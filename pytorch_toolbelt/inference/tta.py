"""Implementation of GPU-friendly test-time augmentation for image segmentation and classification tasks.

Despite this is called test-time augmentation, these method can be used at training time as well since all
transformation written in PyTorch and respect gradients flow.
"""

from torch import Tensor, nn
from . import functional as F


def tta_fliplr_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    for input image and vertically flipped one.

    :param model:
    :param image:
    :return:
    """
    output = model(image) + model(F.torch_fliplp(image))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def tta_fliplr_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image) + F.torch_fliplp(model(F.torch_fliplp(image)))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def tta_d4_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug in [F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    image = F.torch_transpose(image)

    for aug in [F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8


def tta_d4_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug, deaug in zip([F.torch_rot90, F.torch_rot180, F.torch_rot270], [F.torch_rot270, F.torch_rot180, F.torch_rot90]):
        x = deaug(model(aug(image)))
        output = output + x

    image = F.torch_transpose(image)

    for aug, deaug in zip([F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270], [F.torch_none, F.torch_rot270, F.torch_rot180, F.torch_rot90]):
        x = deaug(model(aug(image)))
        output = output + x

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8
