# Pytorch-toolbelt

[![Build Status](https://travis-ci.org/BloodAxe/pytorch-toolbelt.svg?branch=develop)](https://travis-ci.org/BloodAxe/pytorch-toolbelt)
[![Documentation Status](https://readthedocs.org/projects/pytorch-toolbelt/badge/?version=latest)](https://pytorch-toolbelt.readthedocs.io/en/latest/?badge=latest)


A `pytorch-toolbelt` is a Python library with a set of bells and whistles for PyTorch for fast R&D prototyping and Kaggle farming:

* Easy model building using flexible encoder-decoder architecture.
* Modules: CoordConv, SCSE, Hypercolumn, Depthwise separable convolution and more
* GPU-friendly test-time augmentation TTA for segmentation and classification
* GPU-friendly inference on huge (5000x5000) images
* Every-day common routines (fix/restore random seed, filesystem utils, metrics)
* Fancy losses: Focal, Lovasz, Jaccard and Dice losses, Wing Loss

# Installation

`pip install pytorch_toolbelt`

# Showcase

## Encoder-decoder models construction

```python
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D

class FPNSegmentationModel(nn.Module):
    def __init__(self, encoder:E.EncoderModule, num_classes, fpn_features=128):
        self.encoder = encoder
        self.decoder = D.FPNDecoder(encoder.output_filters, fpn_features=fpn_features)
        self.fuse = D.FPNFuse()
        input_channels = sum(self.decoder.output_filters)
        self.logits = nn.Conv2d(input_channels, num_classes,kernel_size=1)
        
    def forward(self, input):
        features = self.encoder(input)
        features = self.decoder(features)
        features = self.fuse(features)
        logits = self.logits(features)
        return logits
        
def fpn_resnext50(num_classes):
  encoder = E.SEResNeXt50Encoder()
  return FPNSegmentationModel(encoder, num_classes)
  
def fpn_mobilenet(num_classes):
  encoder = E.MobilenetV2Encoder()
  return FPNSegmentationModel(encoder, num_classes)
```

## Compose multiple losses

```python
from pytorch_toolbelt import losses as L

loss = L.JointLoss(L.FocalLoss(), 1.0, L.LovaszLoss(), 0.5)
```

## Test-time augmentation

```python
from pytorch_toolbelt.inference import tta

# Truly functional TTA for image classification using horizontal flips:
logits = tta.fliplr_image2label(model, input)

# Truly functional TTA for image segmentation using D4 augmentation:
logits = tta.d4_image2mask(model, input)

# TTA using wrapper module:
tta_model = tta.TTAWrapper(model, tta.fivecrop_image2label, crop_size=512)
logits = tta_model(input)
```

## Inference on huge images:

```python
import numpy as np
import torch
import cv2

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy


image = cv2.imread('really_huge_image.jpg')
model = get_model(...)

# Cut large image into overlapping tiles
tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256), weight='pyramid')

# HCW -> CHW. Optionally, do normalization here
tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(image)]

# Allocate a CUDA buffer for holding entire mask
merger = CudaTileMerger(tiler.target_shape, 1, tiler.weight)

# Run predictions for tiles and accumulate them
for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=8, pin_memory=True):
    tiles_batch = tiles_batch.float().cuda()
    pred_batch = model(tiles_batch)

    merger.integrate_batch(pred_batch, coords_batch)

# Normalize accumulated mask and convert back to numpy
merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
merged_mask = tiler.crop_to_orignal_size(merged_mask)
```