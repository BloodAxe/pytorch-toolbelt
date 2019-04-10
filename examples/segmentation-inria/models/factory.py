from torch import nn

from .fpn import fpn_resnext50
from .unet import UNet
from .linknet import LinkNet152, LinkNet34


def get_model(model_name: str, image_size=None) -> nn.Module:
    if model_name == 'unet':
        return UNet()

    if model_name == 'fpn_resnext50':
        return fpn_resnext50()

    if model_name == 'linknet34':
        return LinkNet34()

    if model_name == 'linknet152':
        return LinkNet152()

    raise ValueError("Unsupported model name " + model_name)
