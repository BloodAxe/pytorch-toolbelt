"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""
from .common import *
from .densenet import *
from .efficientnet import *
from .hrnet import *
from .inception import *
from .mobilenet import *
from .resnet import *
from .seresnet import *
from .squeezenet import *
from .unet import *
from .wide_resnet import *
from .xresnet import *
from .hourglass import *
from .timm import *
