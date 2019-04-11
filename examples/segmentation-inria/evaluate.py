import argparse
import os

import cv2
import numpy as np
import torch
from catalyst.dl.utils import UtilsFactory
from models.factory import get_model, predict
from tqdm import tqdm

from pytorch_toolbelt.utils.fs import auto_file, find_in_dir, read_rgb_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='unet', help='')
    parser.add_argument('-dd', '--data-dir', type=str, default=None, required=True, help='Data dir')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, required=True, help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('-tta', '--tta', default=None, type=str, help='Type of TTA to use [fliplr, d4]')
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_file = auto_file(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_file))
    out_dir = os.path.join(run_dir, 'evaluation')
    os.makedirs(out_dir, exist_ok=True)

    model = get_model(args.model)

    checkpoint = UtilsFactory.load_checkpoint(checkpoint_file)
    checkpoint_epoch = checkpoint['epoch']
    print('Loaded model weights from', args.checkpoint)
    print('Epoch   :', checkpoint_epoch)
    print('Metrics (Train):', 'IoU:', checkpoint['epoch_metrics']['train']['jaccard'], 'Acc:', checkpoint['epoch_metrics']['train']['accuracy'])
    print('Metrics (Valid):', 'IoU:', checkpoint['epoch_metrics']['valid']['jaccard'], 'Acc:', checkpoint['epoch_metrics']['valid']['accuracy'])

    UtilsFactory.unpack_checkpoint(checkpoint, model=model)

    model = model.cuda().eval()

    train_images = find_in_dir(os.path.join(data_dir, 'train', 'images'))
    for fname in tqdm(train_images, total=len(train_images)):
        image = read_rgb_image(fname)
        mask = predict(model, image, tta=args.tta, image_size=(512, 512), batch_size=args.batch_size, activation='sigmoid')
        mask = (mask * 255).astype(np.uint8)
        name = os.path.join(out_dir, os.path.basename(fname))
        cv2.imwrite(name, mask)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
