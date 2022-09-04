from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict, Type

from pytorch_toolbelt.datasets import OUTPUT_MASK_KEY
from pytorch_toolbelt.modules import (
    conv1x1,
    UnetBlock,
    ACT_RELU,
    ABN,
    ResidualDeconvolutionUpsample2d,
    ACT_SILU,
)
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn, Tensor
from torch.nn import functional as F

__all__ = [
    "UnetSegmentationModel",
    "b6_unet32_s2_rdtc",
]


class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=ACT_RELU,
        upsample_block: Union[Type[nn.Upsample], Type[ResidualDeconvolutionUpsample2d]] = nn.UpsamplingNearest2d,
        last_upsample_block=None,
        unet_block=UnetBlock,
        abn_block=ABN,
    ):
        super().__init__()
        self.encoder = encoder

        abn_block = partial(abn_block, activation=activation)
        self.decoder = UNetDecoder(
            feature_maps=encoder.channels,
            decoder_features=unet_channels,
            unet_block=partial(unet_block, abn_block=abn_block),
            upsample_block=upsample_block,
        )

        if last_upsample_block is not None:
            self.last_upsample_block = last_upsample_block(unet_channels[0])
            self.mask = nn.Sequential(
                OrderedDict(
                    [
                        ("drop", nn.Dropout2d(dropout)),
                        ("conv", conv1x1(self.last_upsample_block.out_channels, num_classes)),
                    ]
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
        if self.last_upsample_block is not None:
            mask = self.mask(self.last_upsample_block(x[0]))
        else:
            mask = self.mask(x[0])
            if self.full_size_mask:
                mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}
        return output


def b6_unet32_s2_rdtc(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.B6Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    from pytorch_toolbelt.modules.upsample import ResidualDeconvolutionUpsample2d

    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[32, 64, 128, 256],
        activation=ACT_SILU,
        dropout=dropout,
        upsample_block=ResidualDeconvolutionUpsample2d,
        last_upsample_block=ResidualDeconvolutionUpsample2d,
    )
