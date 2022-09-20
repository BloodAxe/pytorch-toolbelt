import collections
import os
import sys
from datetime import datetime
from functools import partial
from time import sleep
from typing import Tuple, Any, Optional, Dict, List, OrderedDict, Union, Callable

import cv2
import hydra.utils
import numpy as np
import pytorch_toolbelt.utils.catalyst
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
from painless_sota.inria_aerial.callbacks import JaccardMetricPerImageWithOptimalThreshold
from painless_sota.inria_aerial.factory import get_scheduler, get_optimizer
from pytorch_toolbelt.datasets import *
from pytorch_toolbelt.datasets.providers.inria_aerial import InriaAerialImageDataset
from pytorch_toolbelt.optimization.functional import freeze_model
from pytorch_toolbelt.utils.catalyst.visualization import draw_binary_segmentation_predictions

from pytorch_toolbelt.utils import (
    fs,
    transfer_weights,
    count_parameters,
    master_print,
    is_dist_avail_and_initialized,
    get_collate_for_dataset,
    rgb_image_from_tensor,
    to_numpy,
    hstack_autopad,
)
from pytorch_toolbelt.utils.catalyst.pipeline import (
    scale_learning_rate_for_ddp,
)
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DistributedSampler, Sampler, DataLoader, Dataset

__all__ = ["InriaAerialPipeline"]


class InriaAerialPipeline:
    _distributed = False
    _local_rank = 0
    _world_size = 1
    _is_master = True
    config: DictConfig
    experiment_dir: Optional[str]

    def __init__(self, config: DictConfig):
        self.config = config
        self.discover_distributed_params()
        self.experiment_dir = self.get_experiment_dir(config)

    def train(self):
        config = self.config

        model = self.build_model(config)
        loaders, dataset_callbacks = self.build_loaders(config)
        optimizer, scheduler, optimizer_callbacks = self.build_optimizer(model, loaders)
        criterions, criterions_callbacks = self.build_criterions(config)
        metric_callbacks = self.build_metrics(config, loaders, model)

        experiment_dir = self.experiment_dir

        if self.is_master:
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
                OmegaConf.save(self.config, f, resolve=False)
            with open(os.path.join(experiment_dir, "config_resolved.yaml"), "w") as f:
                OmegaConf.save(self.config, f, resolve=True)

        # model training
        input_key = config.model.input_key
        output_key = config.model.output_key

        if input_key is not None and not isinstance(input_key, str):
            input_key = OmegaConf.to_container(input_key, resolve=True)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key, resolve=True)

        runner = SupervisedRunner(input_key=input_key, output_key=output_key, device="cuda")
        extra_callbacks = [TimerCallback()]
        callbacks = dataset_callbacks + metric_callbacks + optimizer_callbacks + criterions_callbacks + extra_callbacks

        runner.train(
            fp16=self.build_distributed_params(config),
            model=model,
            criterion=criterions,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=experiment_dir,
            num_epochs=config.runner.max_epochs,
            verbose=True,
            main_metric=config.runner.main_metric,
            minimize_metric=config.runner.main_metric_minimize,
            checkpoint_data=self.get_checkpoint_data(),
        )

    @classmethod
    def get_experiment_dir(cls, config: DictConfig) -> str:
        if config.experiment:
            return config.experiment

        current_time = datetime.now().strftime("%y%m%d_%H_%M")
        hc = hydra.utils.HydraConfig.get()
        experiment_slug = (
            f"{current_time}_{hc.runtime.choices.dataset}_{hc.runtime.choices.model}_{hc.runtime.choices.loss}"
        )

        fold = config["dataset"].get("fold", None)
        if fold is not None:
            experiment_slug += f"_fold{fold}"

        log_dir = os.path.join("runs", experiment_slug)
        return log_dir

    def get_checkpoint_data(self):
        return {
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

    def discover_distributed_params(self):
        self._local_rank = int(self.config.local_rank)
        self._world_size = int(self.config.world_size)
        self._distributed = int(self.config.world_size) > 1
        if not self._distributed:
            return
        sleep(int(self.config.world_size) * 0.1)
        print("Initializing init_process_group", self.config.local_rank, flush=True)
        torch.cuda.set_device(int(self.config.local_rank))
        torch.distributed.init_process_group(backend="nccl")

        sleep(int(self.config.world_size) * 0.1)
        print("Initialized init_process_group", int(self.config.local_rank), flush=True)

        self._is_master = (int(self.config.local_rank) == 0) | (not self._distributed)

        try:
            if self.distributed and self.is_linux and self.config.torch.use_affinity_mask:
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

    def build_datasets(self, config) -> Tuple[Dataset, Dataset, Optional[Sampler], List[Callback]]:
        provider: InriaAerialImageDataset = instantiate(config.dataset.provider)
        full_df = provider.get_train_val_split_train_df()
        train_df = full_df[full_df.split == "train"]
        valid_df = full_df[full_df.split == "valid"]

        train_ds, train_sampler = instantiate(config.dataset.train.dataset, df=train_df)
        valid_ds = instantiate(config.dataset.validation.dataset, df=valid_df)

        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        master_print("Train dataset", len(train_ds))
        if train_sampler is not None:
            master_print("Train sampler", train_sampler.num_samples)
        master_print("Valid dataset", len(valid_ds))

        return train_ds, valid_ds, train_sampler, []

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
            batch_size=config.dataset.train.loader.batch_size,
            num_workers=config.dataset.train.loader.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=get_collate_for_dataset(train_ds),
            persistent_workers=persistent_workers,
        )
        master_print("Train loader")
        master_print("  Batch Size ", loaders["train"].batch_size)
        master_print("  Num Batches", len(loaders["train"]))
        master_print("  Num Workers", loaders["train"].num_workers)

        loaders["valid"] = DataLoader(
            valid_ds,
            batch_size=config.dataset.validation.loader.batch_size,
            num_workers=config.dataset.validation.loader.num_workers,
            pin_memory=True,
            sampler=valid_sampler,
            drop_last=False,
            collate_fn=get_collate_for_dataset(valid_ds),
            persistent_workers=persistent_workers,
        )
        master_print("Valid loader")
        master_print("  Length    ", len(loaders["valid"]))
        master_print("  Num Batches", len(loaders["valid"]))
        master_print("  Num Workers", loaders["valid"].num_workers)

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
        self,
        model: nn.Module,
        loaders: OrderedDict,
    ) -> Tuple[Optimizer, Any, List[Callback]]:
        use_fp16 = bool(self.config.optimizer.fp16)
        use_decay_on_bias = bool(self.config.optimizer.wd_on_bias)
        accumulation_steps = int(self.config.optimizer.get("accumulation", 1))

        optimizer_config = self.config.optimizer.optimizer

        optimizer_params = optimizer_config["params"]
        optimizer_params = scale_learning_rate_for_ddp(optimizer_params)

        optimizer = get_optimizer(
            model=model,
            optimizer_name=optimizer_config.name,
            optimizer_params=optimizer_params,
            apply_weight_decay_to_bias=use_decay_on_bias,
            layerwise_params=optimizer_config.get("layerwise_params", None),
        )

        if use_fp16:
            opt_callback = AMPOptimizerCallback(accumulation_steps=accumulation_steps)
        else:
            opt_callback = OptimizerCallback(accumulation_steps=accumulation_steps, decouple_weight_decay=False)
        callbacks = [opt_callback]
        master_print("Optimizer        :", optimizer_config.name)
        master_print("  Parameters     :", count_parameters(model, human_friendly=True))
        master_print("  FP16           :", use_fp16)
        master_print("  Learning rate  :", optimizer_params["lr"])
        master_print("  Weight decay   :", optimizer_params.get("weight_decay", 0))
        master_print("  WD on bias     :", use_decay_on_bias)
        master_print("  Accumulation   :", accumulation_steps)
        master_print("Params           :")
        for k, v in optimizer_params.items():
            master_print(f"  {k}:", v)

        scheduler_params = self.config["optimizer"]["scheduler"]
        if scheduler_params is not None:
            num_epochs = self.config.runner.max_epochs

            scheduler = get_scheduler(
                optimizer,
                learning_rate=optimizer_params["lr"],
                num_epochs=num_epochs,
                batches_in_epoch=len(loaders["train"]),
                **scheduler_params,
            )

            if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
                callbacks += [SchedulerCallback(mode="batch")]
            else:
                callbacks += [SchedulerCallback(mode="epoch")]

            master_print("Scheduler        :", scheduler_params.name)
        else:
            scheduler = None

        return optimizer, scheduler, callbacks

    def build_model(self, config: DictConfig) -> nn.Module:
        model = instantiate(config.model.architecture, _recursive_=False)

        if self.config.transfer:
            transfer_checkpoint = fs.auto_file(self.config.transfer)
            master_print("Transferring weights from model checkpoint", transfer_checkpoint)
            checkpoint = load_checkpoint(transfer_checkpoint)
            pretrained_dict = checkpoint["model_state_dict"]

            transfer_weights(model, pretrained_dict)
        elif self.config.checkpoint:
            checkpoint = load_checkpoint(fs.auto_file(self.config.checkpoint))
            unpack_checkpoint(checkpoint, model=model)

            master_print("Loaded model weights from:", self.config.checkpoint)
            pytorch_toolbelt.utils.catalyst.report_checkpoint(checkpoint)

        model = model.cuda()
        if self.config.torch.channels_last:
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
    ) -> Tuple[Dict, CriterionCallback, str]:
        if target_key is not None and not isinstance(target_key, str):
            target_key = OmegaConf.to_container(target_key)
        if output_key is not None and not isinstance(output_key, str):
            output_key = OmegaConf.to_container(output_key)

        criterion: nn.Module = hydra.utils.instantiate(loss_config, **{"_convert_": "all", "_recursive_": True})

        if not isinstance(criterion, nn.Module):
            raise RuntimeError("Loss module must be subclass of nn.Module")
        criterion = criterion.cuda()
        criterions_dict = {f"{prefix}": criterion}

        criterion_callback = CriterionCallback(
            prefix=f"{prefix}",
            input_key=target_key,
            output_key=output_key,
            criterion_key=f"{prefix}",
            multiplier=float(loss_weight),
        )

        return criterions_dict, criterion_callback, prefix

    def build_criterions(self, config: DictConfig) -> Tuple[Dict[str, nn.Module], List[Callback]]:
        losses = []
        criterions_dict = {}
        callbacks = []

        losses_aggregation = config["loss"]["aggregation"]
        criterions_config: List[Dict] = config["loss"]["losses"]

        master_print("Losses")

        for criterion_cfg in criterions_config:
            loss_weight = criterion_cfg.get("weight", 1.0)

            criterion_loss, criterion_callback, criterion_name = self.get_criterion_callback(
                criterion_cfg["loss"],
                prefix="losses/" + criterion_cfg["prefix"],
                target_key=criterion_cfg["target_key"],
                output_key=criterion_cfg["output_key"],
                loss_weight=float(loss_weight),
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

    def build_metrics(self, config, loaders, model):
        callbacks = [
            JaccardMetricPerImageWithOptimalThreshold(
                prefix=config.runner.main_metric,
                predictions_postprocess_fn=torch.sigmoid,
                predictions_key=OUTPUT_MASK_KEY,
                targets_key=TARGET_MASK_KEY,
                image_id_key=INPUT_IMAGE_ID_KEY,
            ),
        ]

        if self.is_master:
            callbacks += [
                pytorch_toolbelt.utils.catalyst.BestMetricCheckpointCallback(
                    target_metric=config.runner.main_metric,
                    target_metric_minimize=config.runner.main_metric_minimize,
                    save_n_best=3,
                ),
            ]

            show_batches = config.runner.get("show_batches", False)
            if show_batches:

                visualize_fn = partial(
                    draw_inria_segmentation_predictions,
                    image_key=INPUT_IMAGE_KEY,
                    image_id_key=INPUT_IMAGE_ID_KEY,
                    targets_key=TARGET_MASK_KEY,
                    outputs_key=OUTPUT_MASK_KEY,
                    max_images=None,
                    targets_threshold=0.5,
                    logits_threshold=0,
                )
                callbacks += [
                    pytorch_toolbelt.utils.catalyst.ShowPolarBatchesCallback(
                        visualize_fn, metric="loss", minimize=True
                    ),
                ]

        if config.runner.early_stopping > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    metric=config.runner.main_metric,
                    minimize=config.runner.main_metric_minimize,
                    min_delta=1e-6,
                    patience=config.runner.early_stopping,
                )
            )

        return callbacks


def draw_inria_segmentation_predictions(
    input: dict,
    output: dict,
    image_key="features",
    image_id_key: Optional[str] = "image_id",
    targets_key="targets",
    outputs_key="logits",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=None,
    targets_threshold=0.5,
    logits_threshold=0,
    image_format: Union[str, Callable] = "rgb",
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives

    :param input: Input batch (model's input batch)
    :param output: Output batch (model predictions)
    :param image_key: Key for getting image
    :param image_id_key: Key for getting image id/fname
    :param targets_key: Key for getting ground-truth mask
    :param outputs_key: Key for getting model logits for predicted mask
    :param mean: Mean vector user during normalization
    :param std: Std vector user during normalization
    :param max_images: Maximum number of images to visualize from batch
        (If you have huge batch, saving hundreds of images may make TensorBoard slow)
    :param targets_threshold: Threshold to convert target values to binary.
        Default value 0.5 is safe for both smoothed and hard labels.
    :param logits_threshold: Threshold to convert model predictions (raw logits) values to binary.
        Default value 0.0 is equivalent to 0.5 after applying sigmoid activation
    :param image_format: Source format of the image tensor to conver to RGB representation.
        Can be string ("gray", "rgb", "brg") or function `convert(np.ndarray)->nd.ndarray`.
    :return: List of images
    """
    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    assert output[outputs_key].size(1) == 1, "Mask must be single-channel tensor of shape [Nx1xHxW]"

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)

        if image_format == "rgb":
            pass
        elif image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_format, callable):
            image = image_format(image)

        target = to_numpy(input[targets_key][i]).squeeze(0)
        logits = to_numpy(output[outputs_key][i]).squeeze(0)

        overlay = image.copy()
        true_mask = target > targets_threshold
        pred_mask = logits > logits_threshold

        overlay[true_mask & pred_mask] = np.array(
            [0, 250, 0], dtype=overlay.dtype
        )  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array(
            [250, 250, 0], dtype=overlay.dtype
        )  # False alarm painted with yellow
        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)

        if OUTPUT_MASK_KEY_STRIDE_4 in output and TARGET_MASK_KEY_STRIDE_4 in input:
            target = to_numpy(input[TARGET_MASK_KEY_STRIDE_4][i]).squeeze(0)
            logits = to_numpy(output[OUTPUT_MASK_KEY_STRIDE_4][i]).squeeze(0)

            overlay2 = image.copy()
            true_mask = (
                cv2.resize(
                    target, dst=None, dsize=(overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                > targets_threshold
            )
            pred_mask = (
                cv2.resize(
                    logits, dst=None, dsize=(overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                > logits_threshold
            )

            overlay2[true_mask & pred_mask] = np.array(
                [0, 250, 0], dtype=overlay2.dtype
            )  # Correct predictions (Hits) painted with green
            overlay2[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay2.dtype)  # Misses painted with red
            overlay2[~true_mask & pred_mask] = np.array(
                [250, 250, 0], dtype=overlay2.dtype
            )  # False alarm painted with yellow
            overlay2 = cv2.addWeighted(image, 0.5, overlay2, 0.5, 0, dtype=cv2.CV_8U)

            overlay = hstack_autopad([overlay, overlay2])

        if image_id_key is not None and image_id_key in input:
            image_id = input[image_id_key][i]
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images
