from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict

from pytorch_toolbelt.modules import conv1x1, UnetBlock, ACT_RELU, ABN, ACT_SWISH
from pytorch_toolbelt.modules import encoders as E, DepthToSpaceUpsample2d
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn, Tensor
from torch.nn import functional as F


__all__ = ["UnetSegmentationModel", "resnet34_unet32_s2", "resnet34_unet64_s4", "hrnet34_unet64"]


class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=ACT_RELU,
        upsample_block=nn.UpsamplingNearest2d,
        last_upsample_block=None,
    ):
        super().__init__()
        self.encoder = encoder

        abn_block = partial(ABN, activation=activation)
        self.decoder = UNetDecoder(
            feature_maps=encoder.channels,
            decoder_features=unet_channels,
            unet_block=partial(UnetBlock, abn_block=abn_block),
            upsample_block=upsample_block,
        )

        if last_upsample_block is not None:
            self.mask = nn.Sequential(
                OrderedDict(
                    [("drop", nn.Dropout2d(dropout)), ("conv", last_upsample_block(unet_channels[0], num_classes))]
                )
            )
        else:
            self.last_upsample_block = None
            self.mask = nn.Sequential(
                OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(unet_channels[0], num_classes))])
            )

        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])
        if self.full_size_mask and mask.size()[2:] != x_size:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        return mask


def resnet34_unet32_s2(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        activation=ACT_SWISH,
        dropout=dropout,
    )


def resnet34_unet64_s4(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[64, 128, 256],
        activation=ACT_SWISH,
        dropout=dropout,
    )


def hrnet34_unet64(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[64, 128, 256  ], dropout=dropout)
