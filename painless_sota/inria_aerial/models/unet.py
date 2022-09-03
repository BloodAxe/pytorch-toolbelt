from typing import List

from omegaconf import DictConfig
from painless_sota.inria_aerial.models.unet_blocks import get_unet_block
from pytorch_toolbelt.modules import (
    ACT_RELU,
    decoders as D,
    EncoderModule,
    ResidualDeconvolutionUpsample2d,
)
from torch import nn

__all__ = ["SegmentationUNet"]


class SegmentationUNet(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        decoder_features: List[int],
        segmentation_head: nn.Module,
        activation=ACT_RELU,
        block_type="UnetBlock",
        upsample_type="bilinear",
    ):
        super().__init__()
        unet_block = get_unet_block(block_type, activation=activation)
        self.output_stride = encoder.strides[0]
        self.encoder = encoder

        upsample_block = {
            "nearest": nn.UpsamplingNearest2d,
            "bilinear": nn.UpsamplingBilinear2d,
            "rdtsc": ResidualDeconvolutionUpsample2d,
            "shuffle": nn.PixelShuffle,
        }[upsample_type]

        encoder_channels = list(encoder.channels)
        self.decoder = D.UNetDecoder(
            encoder_channels,
            decoder_features,
            unet_block=unet_block,
            upsample_block=upsample_block,
        )
        self.head = segmentation_head

    def forward(self, x):
        output_size = x.size()[2:]
        feature_maps = self.encoder(x)
        feature_maps = self.decoder(feature_maps)
        segmentation_outputs = self.head(feature_maps, output_size=output_size)
        return segmentation_outputs

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder: EncoderModule = instantiate(config.encoder)
        if config.num_channels != 3:
            encoder = encoder.change_input_channels(config.num_channels)

        segmentation_head = instantiate(config.segmentation)

        return SegmentationUNet(
            encoder=encoder,
            decoder_features=config.decoder.channels,
            segmentation_head=segmentation_head,
            activation=config.activation,
            block_type=config.decoder.get("block_type", "UnetBlock"),
            upsample_type=config.decoder.get("upsample_type", "bilinear"),
        )
