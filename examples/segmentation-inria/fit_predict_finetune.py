from __future__ import absolute_import

import argparse
import collections
import os
from datetime import datetime

import torch
from catalyst.dl.callbacks import UtilsFactory, OneCycleLR
from catalyst.dl.experiments import SupervisedRunner
from torch.backends import cudnn

from pytorch_toolbelt.optimization.functional import get_lr_decay_parameters
from pytorch_toolbelt.utils.catalyst_utils import ShowPolarBatchesCallback, EpochJaccardMetric, PixelAccuracyMetric
from pytorch_toolbelt.utils.fs import auto_file
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

from models.factory import get_model, get_loss, get_optimizer, get_dataloaders, visualize_inria_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-dd', '--data-dir', type=str, required=True, help='Data directory for INRIA sattelite dataset')
    parser.add_argument('-m', '--model', type=str, default='unet', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    # parser.add_argument('-f', '--fold', default=None, required=True, type=int, help='Fold to train')
    #     # parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    #     # parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-l', '--criterion', type=str, default='bce', help='Criterion')
    parser.add_argument('-o', '--optimizer', default='SGD', help='Name of the optimizer')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-w', '--workers', default=8, type=int, help='Num workers')

    args = parser.parse_args()
    set_manual_seed(args.seed)

    data_dir = args.data_dir
    num_workers = args.workers
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_name = args.model
    optimizer_name = args.optimizer
    image_size = (512, 512)

    train_loader, valid_loader = get_dataloaders(data_dir=data_dir,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 image_size=image_size,
                                                 fast=args.fast)

    model = maybe_cuda(get_model(model_name, image_size=image_size))
    criterion = get_loss(args.criterion)

    parameters = get_lr_decay_parameters(model.named_parameters(), learning_rate, {
        'encoder.layer0': 0.1,
        'encoder.layer1': 0.2,
        'encoder.layer2': 0.3,
        'encoder.layer3': 0.4,
        'encoder.layer4': 0.5,
        'decoder': 0.75
    })

    optimizer = get_optimizer(optimizer_name, parameters, learning_rate)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5)

    # model runner
    runner = SupervisedRunner()

    if args.checkpoint:
        checkpoint = UtilsFactory.load_checkpoint(auto_file(args.checkpoint))
        UtilsFactory.unpack_checkpoint(checkpoint, model=model)

        checkpoint_epoch = checkpoint['epoch']
        print('Loaded model weights from', args.checkpoint)
        print('Epoch   :', checkpoint_epoch)
        print('Metrics:', checkpoint['epoch_metrics'])

        # try:
        #     UtilsFactory.unpack_checkpoint(checkpoint, optimizer=optimizer)
        # except Exception as e:
        #     print('Failed to restore optimizer state', e)
        #
        # try:
        #     UtilsFactory.unpack_checkpoint(checkpoint, scheduler=scheduler)
        # except Exception as e:
        #     print('Failed to restore scheduler state', e)

        print('Loaded model weights from', args.checkpoint)

    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_{args.model}_finetune'
    log_dir = os.path.join('runs', prefix)
    os.makedirs(log_dir, exist_ok=False)

    print('Train session:', prefix)
    print('\tFast mode  :', args.fast)
    print('\tEpochs     :', num_epochs)
    print('\tWorkers    :', num_workers)
    print('\tData dir   :', data_dir)
    print('\tLog dir    :', log_dir)
    print('\tTrain size :', len(train_loader), len(train_loader.dataset))
    print('\tValid size :', len(valid_loader), len(valid_loader.dataset))
    print('Model:', model_name)
    print('\tParameters:', count_parameters(model))
    print('\tImage size:', image_size)
    print('Optimizer:', optimizer_name)
    print('\tLearning rate:', learning_rate)
    print('\tBatch size   :', batch_size)
    print('\tCriterion    :', args.criterion)

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        callbacks=[
            OneCycleLR(
                cycle_len=num_epochs,
                div_factor=10,
                increase_fraction=0.3,
                momentum_range=(0.95, 0.85)),
            PixelAccuracyMetric(),
            EpochJaccardMetric(),
            ShowPolarBatchesCallback(visualize_inria_predictions, metric='accuracy', minimize=False),
        ],
        loaders=loaders,
        logdir=log_dir,
        num_epochs=num_epochs,
        verbose=True,
        main_metric='jaccard',
        minimize_metric=False,
        state_kwargs={"cmd_args": vars(args)}
    )

    # Training is finished. Let's run predictions using best checkpointing weights
    # best_checkpoint = UtilsFactory.load_checkpoint(auto_file('best.pth', where=log_dir))
    # UtilsFactory.unpack_checkpoint(best_checkpoint, model=model)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
