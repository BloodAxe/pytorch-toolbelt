# Important Update

![ukraine-flag](docs/480px-Flag_of_Ukraine.jpg)

On February 24th, 2022, Russia declared war and invaded peaceful Ukraine. 
After the annexation of Crimea and the occupation of the Donbas region, Putin's regime decided to destroy Ukrainian nationality.
Ukrainians show fierce resistance and demonstrate to the entire world what it's like to fight for the nation's independence.

Ukraine's government launched a website to help russian mothers, wives & sisters find their beloved ones killed or captured in Ukraine - https://200rf.com & https://t.me/rf200_now (Telegram channel).
Our goal is to inform those still in Russia & Belarus, so they refuse to assault Ukraine. 

Help us get maximum exposure to what is happening in Ukraine, violence, and inhuman acts of terror that the "Russian World" has brought to Ukraine. 
This is a comprehensive Wiki on how you can help end this war: https://how-to-help-ukraine-now.super.site/ 

Official channels
* [Official account of the Parliament of Ukraine](https://t.me/verkhovnaradaofukraine)
* [Ministry of Defence](https://www.facebook.com/MinistryofDefence.UA)
* [Office of the president](https://www.facebook.com/president.gov.ua)
* [Cabinet of Ministers of Ukraine](https://www.facebook.com/KabminUA)
* [Center of strategic communications](https://www.facebook.com/StratcomCentreUA)
* [Minister of Foreign Affairs of Ukraine](https://twitter.com/DmytroKuleba)

Glory to Ukraine!


# Pytorch-toolbelt

A `pytorch-toolbelt` is a Python library with a set of bells and whistles for PyTorch for fast R&D prototyping and Kaggle farming:

## What's inside

* Easy model building using flexible encoder-decoder architecture.
* Modules: CoordConv, SCSE, Hypercolumn, Depthwise separable convolution and more.
* GPU-friendly test-time augmentation TTA for segmentation and classification
* GPU-friendly inference on huge (5000x5000) images
* Every-day common routines (fix/restore random seed, filesystem utils, metrics)
* Losses: BinaryFocalLoss, Focal, ReducedFocal, Lovasz, Jaccard and Dice losses, Wing Loss and more.
* Extras for [Catalyst](https://github.com/catalyst-team/catalyst) library (Visualization of batch predictions, additional metrics) 

Showcase: [Catalyst, Albumentations, Pytorch Toolbelt example: Semantic Segmentation @ CamVid](https://colab.research.google.com/drive/1OUPJYU7TzH5Vz1si6FBkooackuIlzaGr#scrollTo=GUWuiO5K3aUm)

# Why

Honest answer is "I needed a convenient way to re-use code for my Kaggle career". 
During 2018 I achieved a [Kaggle Master](https://www.kaggle.com/bloodaxe) badge and this been a long path. 
Very often I found myself re-using most of the old pipelines over and over again. 
At some point it crystallized into this repository. 

This lib is not meant to replace catalyst / ignite / fast.ai high-level frameworks. Instead it's designed to complement them.

# Installation

`pip install pytorch_toolbelt`

# How do I ... 

## Model creation

### Create Encoder-Decoder U-Net model

Below a code snippet that creates vanilla U-Net model for binary segmentation. 
By design, both encoder and decoder produces a list of tensors, from fine (high-resolution, indexed `0`) to coarse (low-resolution) feature maps. 
Access to all intermediate feature maps is beneficial if you want to apply deep supervision losses on them or encoder-decoder of object detection task, 
where access to intermediate feature maps is necessary.
 
```python
from torch import nn
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D

class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.encoder = E.UnetEncoder(in_channels=input_channels, out_channels=32, growth_factor=2)
        self.decoder = D.UNetDecoder(self.encoder.channels, decoder_features=32)
        self.logits = nn.Conv2d(self.decoder.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.logits(x[0])
```

### Create Encoder-Decoder FPN model with pretrained encoder

Similarly to previous example, you can change decoder to FPN with contatenation. 

 ```python
from torch import nn
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D

class SEResNeXt50FPN(nn.Module):
    def __init__(self, num_classes, fpn_channels):
        super().__init__()
        self.encoder = E.SEResNeXt50Encoder()
        self.decoder = D.FPNCatDecoder(self.encoder.channels, fpn_channels)
        self.logits = nn.Conv2d(self.decoder.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.logits(x[0])
```

### Change number of input channels for the Encoder

All encoders from `pytorch_toolbelt` supports changing number of input channels. Simply call `encoder.change_input_channels(num_channels)` and first convolution layer will be changed.
Whenever possible, existing weights of convolutional layer will be re-used (in case new number of channels is greater than default, new weight tensor will be padded with randomly-initialized weigths).
Class method returns `self`, so this call can be chained.


```python
from pytorch_toolbelt.modules import encoders as E

encoder = E.SEResnet101Encoder()
encoder = encoder.change_input_channels(6)
```


## Misc


## Count number of parameters in encoder/decoder and other modules

When designing a model and optimizing number of features in neural network, I found it's quite useful to print number of parameters in high-level blocks (like `encoder` and `decoder`).
Here is how to do it with `pytorch_toolbelt`:


```python
from torch import nn
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.utils import count_parameters

class SEResNeXt50FPN(nn.Module):
    def __init__(self, num_classes, fpn_channels):
        super().__init__()
        self.encoder = E.SEResNeXt50Encoder()
        self.decoder = D.FPNCatDecoder(self.encoder.channels, fpn_channels)
        self.logits = nn.Conv2d(self.decoder.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.logits(x[0])

net = SEResNeXt50FPN(1, 128)
print(count_parameters(net))
# Prints {'total': 34232561, 'trainable': 34232561, 'encoder': 25510896, 'decoder': 8721536, 'logits': 129}

```

### Compose multiple losses

There are multiple ways to combine multiple losses, and high-level DL frameworks like Catalyst offers way more flexible way to achieve this, but here's 100%-pure PyTorch implementation of mine:

```python
from pytorch_toolbelt import losses as L

# Creates a loss function that is a weighted sum of focal loss 
# and lovasz loss with weigths 1.0 and 0.5 accordingly.
loss = L.JointLoss(L.FocalLoss(), L.LovaszLoss(), 1.0, 0.5)
```


## TTA / Inferencing

### Apply Test-time augmentation (TTA) for the model

Test-time augmetnation (TTA) can be used in both training and testing phases. 

```python
from pytorch_toolbelt.inference import tta

model = UNet()

# Truly functional TTA for image classification using horizontal flips:
logits = tta.fliplr_image2label(model, input)

# Truly functional TTA for image segmentation using D4 augmentation:
logits = tta.d4_image2mask(model, input)

```

### Inference on huge images:

Quite often, there is a need to perform image segmentation for enormously big image (5000px and more). There are a few problems with such a big pixel arrays:
 1. There are size limitations on maximum size of CUDA tensors (Concrete numbers depends on driver and GPU version)
 2. Heavy CNNs architectures may eat up all available GPU memory with ease when inferencing relatively small 1024x1024 images, leaving no room to bigger image resolution.
  
One of the solutions is to slice input image into tiles (optionally overlapping) and feed each through model and concatenate the results back. 
In this way you can guarantee upper limit of GPU ram usage, while keeping ability to process arbitrary-sized images on GPU.
  

```python
import numpy as np
from torch.utils.data import DataLoader
import cv2

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy


image = cv2.imread('really_huge_image.jpg')
model = get_model(...)

# Cut large image into overlapping tiles
tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256))

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

## Advanced examples

1. [Inria Sattelite Segmentation](https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example)
1. [CamVid Semantic Segmentation](https://github.com/BloodAxe/Catalyst-CamVid-Segmentation-Example)


## Citation

```
@misc{Khvedchenya_Eugene_2019_PyTorch_Toolbelt,
  author = {Khvedchenya, Eugene},
  title = {PyTorch Toolbelt},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BloodAxe/pytorch-toolbelt}},
  commit = {cc5e9973cdb0dcbf1c6b6e1401bf44b9c69e13f3}
}
```
