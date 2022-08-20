import collections
import os
import sys
from datetime import datetime
from functools import partial
from time import sleep
from typing import Tuple, Any, Optional, Dict, List, OrderedDict

import hydra.utils
import numpy as np
import torch
from catalyst.callbacks import (
    MetricAggregationCallback,
    CriterionCallback,
    SchedulerCallback,
    AMPOptimizerCallback,
    OptimizerCallback,
    TimerCallback,
    EarlyStoppingCallback,
)
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst.core import Callback
from catalyst.data import DistributedSamplerWrapper
from catalyst.runners import SupervisedRunner
from catalyst.utils import unpack_checkpoint, load_checkpoint
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DistributedSampler, Sampler, DataLoader, Dataset

from painless_sota.inria_aerial.callbacks import JaccardMetricPerImageWithOptimalThreshold
from pytorch_toolbelt.datasets import *
from pytorch_toolbelt.optimization.functional import freeze_model, get_lr_decay_parameters, get_optimizable_parameters
from pytorch_toolbelt.utils import fs, transfer_weights, count_parameters, master_print, is_dist_avail_and_initialized
from pytorch_toolbelt.utils.catalyst import (
    report_checkpoint,
    MixupCriterionCallback,
    MixupInputCallback,
    HyperParametersCallback,
    ShowPolarBatchesCallback,
    BestMetricCheckpointCallback,
)
from pytorch_toolbelt.utils.catalyst.pipeline import (
    get_optimizer_cls,
    scale_learning_rate_for_ddp,
)

__all__ = ["InriaAerialPipeline"]


class InriaAerialPipeline:
    _distributed = False
    _local_rank = 0
    _world_size = 1
    _is_master = True
    cfg: DictConfig
    experiment_dir: Optional[str]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.discover_distributed_params()
        self.experiment_dir = None

    def get_experiment_dir(self, config) -> str:
        if self.cfg.experiment:
            return self.cfg.experiment

        current_time = datetime.now().strftime("%y%m%d_%H_%M")
        hc = hydra.utils.HydraConfig.get()
        experiment_slug = f"{current_time}_{hc.runtime.choices.model}_{hc.runtime.choices.dataset}_{hc.runtime.choices.loss}_{hc.runtime.choices.augs}"

        fold = config["dataset"].get("fold", None)
        if fold is not None:
            experiment_slug += f"_fold{fold}"

        log_dir = os.path.join("runs", experiment_slug)
        return log_dir

    def build_quircks(self, config: DictConfig) -> List[Callback]:
        extra_callbacks = []

        if config.ema.callback is not None:
            cb = hydra.utils.instantiate(config.ema.callback)
            extra_callbacks.append(cb)
            master_print("Using EMA Callback", cb)

        return extra_callbacks

    def train(self):
        config = self.cfg
        self.on_experiment_start()

        model = self.build_model(config)
        loaders, dataset_callbacks = self.build_loaders(config)
        optimizer, scheduler, optimizer_callbacks = self.build_optimizer(model, loaders, self.cfg.train.tasks)
        criterions, criterions_callbacks = self.build_criterions(config)
        metric_callbacks = self.build_metrics(config, loaders, model)

        experiment_dir = self.get_experiment_dir(config)
        self.experiment_dir = experiment_dir

        if self.is_master:
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
                OmegaConf.save(self.cfg, f, resolve=False)
            with open(os.path.join(experiment_dir, "config_resolved.yaml"), "w") as f:
                OmegaConf.save(self.cfg, f, resolve=True)

        # model training
        runner_config = config.get("runner", {})
        input_key = runner_config.get("input_key", INPUT_IMAGE_KEY)
        output_key = runner_config.get("output_key", None)

        if input_key is not None and not isinstance(input_key, str):
            input_key = OmegaConf.to_container(input_key, resolve=True)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key, resolve=True)

        runner = SupervisedRunner(input_key=input_key, output_key=output_key, device="cuda")
        extra_callbacks = [TimerCallback()]
        # try:
        runner.train(
            fp16=self.build_distributed_params(config),
            model=model,
            criterion=criterions,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=dataset_callbacks
            + metric_callbacks
            + optimizer_callbacks
            + criterions_callbacks
            + extra_callbacks,
            loaders=loaders,
            logdir=experiment_dir,
            num_epochs=config.train.epochs,
            verbose=True,
            main_metric=config.runner.main_metric,
            minimize_metric=config.runner.main_metric_minimize,
            checkpoint_data=self.get_checkpoint_data(),
        )
        # except Exception as e:
        #     with open(os.path.join(experiment_dir, "exception.log"), "w") as logf:
        #         logf.write(str(e))
        #         traceback.print_exc(file=logf)
        #         print(e)

        self.on_experiment_finished()

    def get_checkpoint_data(self):
        return {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

    def discover_distributed_params(self):
        self._local_rank = int(self.cfg.local_rank)
        self._world_size = int(self.cfg.world_size)
        self._distributed = int(self.cfg.world_size) > 1
        if not self._distributed:
            return
        sleep(int(self.cfg.world_size) * 0.1)
        print("Initializing init_process_group", self.cfg.local_rank, flush=True)
        torch.cuda.set_device(int(self.cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")

        sleep(int(self.cfg.world_size) * 0.1)
        print("Initialized init_process_group", int(self.cfg.local_rank), flush=True)

        self._is_master = (int(self.cfg.local_rank) == 0) | (not self._distributed)

        try:
            if self.distributed and self.is_linux and self.cfg.torch.use_affinity_mask:
                import multiprocessing as mp

                pid = 0
                cores = np.split(np.arange(mp.cpu_count()), self._world_size)
                affinity_mask = set(map(int, cores[self._local_rank]))

                print(
                    f"[{self._local_rank}/{self._world_size}] Process is eligible to run on: {os.sched_getaffinity(pid)}"
                )
                os.sched_setaffinity(0, affinity_mask)
                print(
                    f"[{self._local_rank}/{self._world_size}] Now, process is eligible to run on: {os.sched_getaffinity(pid)}"
                )
        except Exception as e:
            print(e)

    @property
    def is_linux(self):
        return sys.platform in {"linux", "linux2"}

    @property
    def distributed(self):
        return self._distributed

    @property
    def is_master(self):
        return self._is_master

    def build_loaders(self, config: DictConfig) -> Tuple[collections.OrderedDict, List[Callback]]:
        train_ds, valid_ds, train_sampler, dataset_callbacks = self.build_datasets(config)
        valid_sampler = None

        if self.distributed:
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
            if train_sampler is not None:
                train_sampler = DistributedSamplerWrapper(train_sampler, world_size, local_rank, shuffle=True)
            else:
                train_sampler = DistributedSampler(train_ds, world_size, local_rank, shuffle=True)
            valid_sampler = DistributedSampler(valid_ds, world_size, local_rank, shuffle=False)

        persistent_workers = sys.platform in {"linux", "linux2"}
        master_print("persistent_workers", persistent_workers)

        loaders = collections.OrderedDict()
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=config.train.loaders.train.batch_size,
            num_workers=config.train.loaders.train.num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=train_ds.get_collate_fn(),
            persistent_workers=persistent_workers,
        )
        master_print("Train loader")
        master_print("  Length    ", len(loaders["train"]))
        master_print("  Batch Size", config.train.loaders.train.batch_size)
        master_print("  Workers   ", config.train.loaders.train.num_workers)

        loaders["valid"] = DataLoader(
            valid_ds,
            batch_size=config.train.loaders.valid.batch_size,
            num_workers=config.train.loaders.valid.num_workers,
            pin_memory=False,
            sampler=valid_sampler,
            drop_last=False,
            collate_fn=valid_ds.get_collate_fn(),
            persistent_workers=persistent_workers,
        )
        master_print("Valid loader")
        master_print("  Length    ", len(loaders["valid"]))
        master_print("  Batch Size", config.train.loaders.valid.batch_size)
        master_print("  Workers   ", config.train.loaders.valid.num_workers)

        return loaders, dataset_callbacks

    def build_distributed_params(self, config: DictConfig) -> Dict:
        fp16 = config.optimizer.get("fp16", False)

        if self.distributed:
            local_rank = torch.distributed.get_rank()
            distributed_params = {"rank": local_rank, "syncbn": True}
            if fp16:
                distributed_params["amp"] = True
            if config.torch.find_unused_parameters:
                distributed_params["find_unused_parameters"] = True
        else:
            if fp16:
                distributed_params = {}
                distributed_params["amp"] = True
            else:
                distributed_params = False

        return distributed_params

    def build_optimizer(
        self, model: nn.Module, loaders: OrderedDict, tasks: List[str]
    ) -> Tuple[Optimizer, Any, List[Callback]]:
        optimizer_config = self.cfg["optimizer"]
        optimizer_use_fp16 = optimizer_config.get("fp16", False)

        optimizer_name = str(optimizer_config["name"])

        wd_on_bias = bool(optimizer_config.get("wd_on_bias", True))
        accumulation_steps = int(optimizer_config.get("accumulation", 1))
        optimizer_params = optimizer_config["params"]
        optimizer_params = scale_learning_rate_for_ddp(optimizer_params)

        optimizer = self.get_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            apply_weight_decay_to_bias=wd_on_bias,
            layerwise_params=optimizer_config.get("layerwise_params", None),
        )

        if optimizer_use_fp16:
            opt_callback = AMPOptimizerCallback(accumulation_steps=accumulation_steps)
        else:
            opt_callback = OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False)

        scheduler_params = self.cfg["scheduler"]
        num_epochs = self.cfg["train"]["epochs"]

        scheduler_name = str(scheduler_params["scheduler_name"])
        scheduler = get_scheduler(
            optimizer,
            learning_rate=optimizer_params["lr"],
            num_epochs=num_epochs,
            batches_in_epoch=len(loaders["train"]),
            **scheduler_params,
        )

        callbacks = [opt_callback]
        if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
            callbacks += [SchedulerCallback(mode="batch")]
        else:
            callbacks += [SchedulerCallback(mode="epoch")]

        master_print("Optimizer        :", optimizer_name)
        # master_print("Model            :", self.cfg.model.config.slug)
        layers = [
            "backbone",
            "rpn",
            "roi_heads",
            "encoder",
            "decoder",
            "head",
            "fuse",
            "extra_stages",
            "center",
            "mask",
        ]

        master_print(
            "  Parameters     :",
            count_parameters(
                model,
                layers,
                human_friendly=True,
            ),
        )
        master_print("  FP16           :", optimizer_use_fp16)
        master_print("  Learning rate  :", optimizer_params["lr"])
        master_print("  Weight decay   :", optimizer_params.get("weight_decay", 0))
        master_print("  WD on bias     :", wd_on_bias)
        master_print("  Accumulation   :", accumulation_steps)
        master_print("Params           :")
        for k, v in optimizer_params.items():
            master_print(f"  {k}:", v)
        master_print("Scheduler        :", scheduler_name)

        return optimizer, scheduler, callbacks

    def get_optimizer(
        self,
        model: nn.Module,
        optimizer_name: str,
        optimizer_params: Dict[str, Any],
        apply_weight_decay_to_bias: bool = True,
        layerwise_params=None,
    ) -> Optimizer:
        """
        Construct an Optimizer for given model
        Args:
            model: Model to optimize. Only parameters that require_grad will be used
            optimizer_name: Name of the optimizer (case-insensitive). Supports native pytorch optimizers, apex and
                optimizers from pytorch-optimizers package.
            optimizer_params: Dict of optimizer params (lr, weight_decay, eps, etc)
            apply_weight_decay_to_bias: Whether to apply weight decay on bias parameters. Default is True
        Returns:
            Optimizer instance
        """

        # Optimizer parameter groups
        if layerwise_params is not None:
            if not apply_weight_decay_to_bias:
                raise ValueError("Layerwise params and no wd on bias are mutually exclusive")

            parameters = get_lr_decay_parameters(model, optimizer_params["lr"], layerwise_params)
        else:
            if apply_weight_decay_to_bias:
                parameters = get_optimizable_parameters(model)
            else:
                default_pg, biases_pg = [], []

                for k, v in model.named_parameters():
                    if v.requires_grad:
                        if str.endswith(k, ".bias"):
                            biases_pg.append(v)  # biases
                        else:
                            default_pg.append(v)  # all else

                if apply_weight_decay_to_bias:
                    parameters = default_pg + biases_pg
                else:
                    parameters = default_pg

        optimizer_cls = get_optimizer_cls(optimizer_name)
        optimizer: Optimizer = optimizer_cls(
            parameters,
            **optimizer_params,
        )

        if not apply_weight_decay_to_bias:
            optimizer.add_param_group({"params": biases_pg, "weight_decay": 0.0})

        return optimizer

    def build_model(self, config: Dict) -> nn.Module:
        train_config: Dict = self.cfg["train"]

        model = self.get_model(config["model"])

        if self.cfg.transfer:
            transfer_checkpoint = fs.auto_file(self.cfg.transfer)
            master_print("Transferring weights from model checkpoint", transfer_checkpoint)
            checkpoint = load_checkpoint(transfer_checkpoint)
            pretrained_dict = checkpoint["model_state_dict"]

            transfer_weights(model, pretrained_dict)
        elif self.cfg.checkpoint:
            checkpoint = load_checkpoint(fs.auto_file(self.cfg.checkpoint))
            unpack_checkpoint(checkpoint, model=model)

            master_print("Loaded model weights from:", self.cfg.checkpoint)
            report_checkpoint(checkpoint)

        freeze_encoder = train_config.get("freeze_encoder", False)
        if freeze_encoder:
            freeze_model(model.encoder, freeze_parameters=True, freeze_bn=False)
            master_print("Frozen model encoder")

        model = model.cuda()
        if self.cfg.torch.channels_last:
            model = model.to(memory_format=torch.channels_last)
            master_print("Using Channels Last")
        return model

    def get_criterion_callback(
        self,
        loss_config,
        target_key,
        output_key,
        prefix: str,
        loss_weight: float = 1.0,
        mixup: bool = False,
        train_only: bool = False,
    ) -> Tuple[Dict, CriterionCallback, str]:
        if target_key is not None and not isinstance(target_key, str):
            target_key = OmegaConf.to_container(target_key)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key)

        criterion = self.get_loss(loss_config)
        if not isinstance(criterion, nn.Module):
            raise RuntimeError("Loss module must be subclass of nn.Module")
        criterion = criterion.cuda()
        criterions_dict = {f"{prefix}": criterion}

        if mixup:
            criterion_callback = MixupCriterionCallback(
                prefix=f"{prefix}",
                input_key=target_key,
                output_key=output_key,
                criterion_key=f"{prefix}",
                multiplier=float(loss_weight),
            )
        # elif train_only:
        #     criterion_callback = TrainOnlyCriterionCallback(
        #         prefix=f"{prefix}",
        #         input_key=target_key,
        #         output_key=output_key,
        #         criterion_key=f"{prefix}",
        #         multiplier=float(loss_weight),
        #     )
        else:
            criterion_callback = CriterionCallback(
                prefix=f"{prefix}",
                input_key=target_key,
                output_key=output_key,
                criterion_key=f"{prefix}",
                multiplier=float(loss_weight),
            )

        return criterions_dict, criterion_callback, prefix

    def build_criterions(self, config: Dict) -> Tuple[Dict[str, nn.Module], List[Callback]]:
        losses = []
        criterions_dict = {}
        callbacks = []

        mixup = config["train"].get("mixup", False)
        losses_aggregation = config["loss"]["aggregation"]
        criterions_config: List[Dict] = config["loss"]["losses"]

        if mixup:
            mixup_a = self.cfg["train"].get("mixup_a", 0.5)
            mixup_p = self.cfg["train"].get("mixup_p", 0.5)
            callbacks.append(
                MixupInputCallback(
                    fields=[INPUT_IMAGE_KEY],
                    alpha=mixup_a,
                    p=mixup_p,
                )
            )
            master_print("Using Mixup", "alpha", mixup_a, "p", mixup_p)

        master_print("Losses")
        train_only = config["loss"].get("train_only", False)
        for criterion_cfg in criterions_config:
            loss_weight = criterion_cfg.get("weight", 1.0)

            criterion_loss, criterion_callback, criterion_name = self.get_criterion_callback(
                criterion_cfg["loss"],
                prefix="losses/" + criterion_cfg["prefix"],
                target_key=criterion_cfg["target_key"],
                output_key=criterion_cfg["output_key"],
                loss_weight=float(loss_weight),
                mixup=mixup,
                train_only=train_only,
            )
            criterions_dict.update(criterion_loss)
            callbacks.append(criterion_callback)
            losses.append(criterion_name)

            master_print(
                "  ",
                criterion_name,
                criterion_loss[criterion_name].__class__.__name__,
                "weight",
                loss_weight,
            )
            master_print(
                "    ",
                "target",
                criterion_cfg["target_key"],
            )
            master_print(
                "    ",
                "output",
                criterion_cfg["output_key"],
            )

        callbacks.append(
            MetricAggregationCallback(
                prefix="loss", metrics=config["loss"].get("losses_weights", losses), mode=losses_aggregation
            )
        )
        return criterions_dict, callbacks

    def get_loss(self, loss_config: Dict):
        return hydra.utils.call(loss_config, **{"_convert_": "all", "_recursive_": True})

    def build_datasets(self, config) -> Tuple[Dataset, Dataset, Optional[Sampler], List[Callback]]:
        augmentations = instantiate(config.augs)
        spatial_augmentations = list(augmentations.spatial)
        color_augmentatinos = list(augmentations.photometric)

        master_print("image_read_function", config.dataset.image_read_function)
        image_read_function = get_image_read_function(config.dataset.image_read_function)

        # validate_shape=True will force loading image to double-check that image shape matches the image shape
        # in the annotation file. Additionally, it will save image as TIFF for faster loading for the image crop.
        if is_main_process():
            instantiate(config.train.datasets.train, read_image_fn=image_read_function, validate_shape=self.is_linux)
            instantiate(
                config.train.datasets.validation, read_image_fn=image_read_function, validate_shape=self.is_linux
            )

        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        self.train_data_provider = instantiate(
            config.train.datasets.train, read_image_fn=image_read_function, validate_shape=False
        )
        self.valid_data_provider = instantiate(
            config.train.datasets.validation, read_image_fn=image_read_function, validate_shape=False
        )

        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        if config.train.copy_paste.enabled:
            copy_paste = CopyPasteAugmentation(
                annotations=self.train_data_provider.annotations,
                objects_to_paste=config.train.copy_paste.objects_to_paste,
                p=config.train.copy_paste.p,
                class_weights="balanced",
                damping=np.sqrt,
                ignored_classes=["ship"],
                read_image_fn=image_read_function,
            )
            color_augmentatinos = [copy_paste] + color_augmentatinos
            master_print("Using copy paste augmentation")

        train_ds, train_sampler = self.train_data_provider.get_dataset_with_sampler(
            crop_size=config.train.train_image_size,
            channels_last=config.torch.channels_last,
            color_augmentations=color_augmentatinos,
            spatial_augmentations=spatial_augmentations,
            min_instance_size=config.dataset.min_instance_size,
            #
            num_samples=config.sampler.num_samples,
            crop_around_center_p=config.sampler.crop_around_center_p,
            balance_crowd=config.sampler.balance_crowd,
            balance_by_type=config.sampler.balance_by_type,
            tasks=self.cfg.train.tasks,
        )

        valid_ds = self.valid_data_provider.get_dataset(
            box_coder=self.box_coder,
            crop_size=config.train.valid_image_size,
            channels_last=config.torch.channels_last,
            min_instance_size=config.dataset.min_instance_size,
            tasks=self.cfg.train.tasks,
        )

        master_print("Train dataset", len(train_ds))
        if train_sampler is not None:
            master_print("Train sampler", train_sampler.num_samples)
        master_print("Valid dataset", len(valid_ds))

        return train_ds, valid_ds, train_sampler, []

    def build_metrics(self, config, loaders, model):
        show_batches = self.cfg["train"].get("show", False)

        callbacks = [
            JaccardMetricPerImageWithOptimalThreshold(
                prefix="metric/jaccard",
                predictions_postprocess_fn=torch.sigmoid,
                predictions_key=OUTPUT_MASK_KEY,
                targets_key=TARGET_MASK_KEY,
            ),
        ]

        if self.is_master:
            callbacks += [
                BestMetricCheckpointCallback(
                    target_metric="det_f05/segmentation",
                    target_metric_minimize=False,
                    save_n_best=3,
                ),
            ]
            if show_batches:
                visualize_fn = partial(
                    visualize_batch,
                )
                callbacks += [
                    ShowPolarBatchesCallback(visualize_fn, metric="loss", minimize=True),
                ]

        if config["train"]["early_stopping"] > 0:
            main_metric = "segmentation"
            callbacks.append(
                EarlyStoppingCallback(
                    metrics=[
                        f"det_f05/{main_metric}",
                    ],
                    minimize=False,
                    min_delta=1e-6,
                    patience=config["train"]["early_stopping"],
                )
            )

        if self.is_master:
            callbacks.append(
                HyperParametersCallback(
                    hparam_dict=dict(
                        optimizer=str(self.cfg.optimizer.name),
                        optimizer_lr=float(self.cfg.optimizer.params.lr),
                        optimizer_eps=float(self.cfg.optimizer.params.eps)
                        if "eps" in self.cfg.optimizer.params
                        else "None",
                        optimizer_wd=float(self.cfg.optimizer.params.weight_decay),
                        optimizer_scheduler=str(self.cfg.scheduler.scheduler_name),
                        # Dataset
                        dataset=str(self.cfg.dataset.slug),
                        dataset_augmentations=str(self.cfg.augs.slug),
                        dataset_train_image_size=f"{self.cfg.train.train_image_size}",
                        # Sampling
                        sampler=str(self.cfg.sampler.slug),
                        sampler_num_samples=int(self.cfg.sampler.num_samples),
                        sampler_type=str(self.cfg.sampler.sampler_type),
                        # Loss
                        loss=str(self.cfg.loss.slug),
                    )
                ),
            )

        return callbacks

    def get_model(self, model_config: DictConfig) -> nn.Module:
        return instantiate(model_config, _recursive_=False)
