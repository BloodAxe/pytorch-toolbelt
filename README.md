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

# Quick start

## Construct model

```python
from pytorch_toolbelt import encoders as E
from pytorch_toolbelt import decoders as D

def fpn_resnext50(num_classes):
  encoder = E.SeResnext50()
  decoder = D.FPN(encoder.output_filters)
  return SegmentationModel(encoder, decoder, num_classes)
```

## Compose multiple losses

```python
from pytorch_toolbelt import loss as L

loss = L.JointLoss(L.FocalLoss(), 1.0, L.LovaszLoss(), 0.5)
```

## Test-time augmentation

```python
model = get_model(...)

# Segmentation case
model = TTAD4(model)

logits = model(batch)
```

## Inference on huge images:

```python
image = cv2.imread('really_huge_image.jpg')
slicer = ImageSlicer(image.shape, tile_size=512, tile_step=256)
merger = CudaTilesMerger(slicer)
tiles = slicer.slice(image)
for batch, coordinates in zip(DataLoader(tiles, batch_size=8, pin_memory=True), slicer.crops):
   logits = model(batch.cuda())
   merger.integrate_batch(logits, coordinates)

mask = merger.merge()
```

# Documentation

`TODO: Implement`
