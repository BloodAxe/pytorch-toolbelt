from torch import nn, Tensor
from torch.nn import functional as F

from ..modules import ABN
from ..modules import decoders as D
from ..modules import encoders as E

__all__ = [
    "FPNSumSegmentationModel",
    "FPNCatSegmentationModel",
    "DeeplabV3SegmentationModel",
    "resnet34_fpncat128",
    "seresnext50_fpncat128",
    "seresnext101_fpncat256",
    "seresnext101_fpnsum256",
    "seresnext101_deeplab256",
    "efficientnetb4_fpncat128",
    "OUTPUT_MASK_KEY",
    "OUTPUT_MASK_4_KEY",
    "OUTPUT_MASK_8_KEY",
    "OUTPUT_MASK_16_KEY",
    "OUTPUT_MASK_32_KEY",
]

OUTPUT_MASK_KEY = "mask"
OUTPUT_MASK_4_KEY = "mask_4"
OUTPUT_MASK_8_KEY = "mask_8"
OUTPUT_MASK_16_KEY = "mask_16"
OUTPUT_MASK_32_KEY = "mask_32"


class FPNSumSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: E.EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        return_full_size_mask=True,
        use_deep_supervision=False,
    ):
        """
        Create a segmentation model of encoder-decoder architecture were encoder is arbitrary architecture
        capable of providing 4 feature maps of 4,8,16,32 stride and decoder is FPN with concatenation.

        Args:
            encoder: Encoder model
            num_classes: Number of channels in final mask
            dropout: Dropout rate to apply before computing final output mask
            abn_block: Activated batch-norm block used in decoder
            fpn_channels: Number of FPN channels computed for each encoder's feature map
            return_full_size_mask: If True, returns mask of same size as input image;
                otherwise returns mask that is 4 times smaller than original image
            use_deep_supervision: If True, model also predicts mask of strides 4, 8, 16, 32 from intermediate layers
                to enforce model learn mask representation at each level of encoder's feature maps
        """
        super().__init__()
        self.encoder = encoder

        self.decoder = D.FPNSumDecoder(
            feature_maps=encoder.output_filters,
            output_channels=num_classes,
            dsv_channels=num_classes if use_deep_supervision else None,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )
        self.deep_supervision = use_deep_supervision
        self.full_size_mask = return_full_size_mask

    def forward(self, x):
        enc_features = self.encoder(x)
        output = self.decoder(enc_features)

        if self.deep_supervision:
            mask, dsv = output
        else:
            mask = output

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.deep_supervision:
            output[OUTPUT_MASK_4_KEY] = dsv[3]
            output[OUTPUT_MASK_8_KEY] = dsv[2]
            output[OUTPUT_MASK_16_KEY] = dsv[1]
            output[OUTPUT_MASK_32_KEY] = dsv[0]

        return output


class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: E.EncoderModule,
        num_classes: int,
        dropout=0.0,
        abn_block=ABN,
        fpn_channels=256,
        return_full_size_mask=True,
        use_deep_supervision=False,
    ):
        """
        Create a segmentation model of encoder-decoder architecture were encoder is arbitrary architecture
        capable of providing 4 feature maps of 4,8,16,32 stride and decoder is FPN with summation.

        Args:
            encoder: Encoder model
            num_classes: Number of channels in final mask
            dropout: Dropout rate to apply before computing final output mask
            abn_block: Activated batch-norm block used in decoder
            fpn_channels: Number of FPN channels computed for each encoder's feature map
            return_full_size_mask: If True, returns mask of same size as input image;
                otherwise returns mask that is 4 times smaller than original image
            use_deep_supervision: If True, model also predicts mask of strides 4, 8, 16, 32 from intermediate layers
                to enforce model learn mask representation at each level of encoder's feature maps
        """
        super().__init__()
        self.encoder = encoder

        self.decoder = D.FPNCatDecoder(
            feature_maps=encoder.output_filters,
            output_channels=num_classes,
            dsv_channels=num_classes if use_deep_supervision else None,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.deep_supervision = use_deep_supervision
        self.full_size_mask = return_full_size_mask

    def forward(self, x: Tensor):
        features = self.encoder(x)
        output = self.decoder(features)

        if self.deep_supervision:
            mask, dsv = output
        else:
            mask = output

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.deep_supervision:
            output[OUTPUT_MASK_4_KEY] = dsv[3]
            output[OUTPUT_MASK_8_KEY] = dsv[2]
            output[OUTPUT_MASK_16_KEY] = dsv[1]
            output[OUTPUT_MASK_32_KEY] = dsv[0]

        return output


class DeeplabV3SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: E.EncoderModule,
        num_classes: int,
        dropout=0.25,
        abn_block=ABN,
        high_level_bottleneck=256,
        low_level_bottleneck=32,
        return_full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = D.DeeplabV3Decoder(
            feature_maps=encoder.output_filters,
            output_stride=encoder.output_strides[-1],
            num_classes=num_classes,
            high_level_bottleneck=high_level_bottleneck,
            low_level_bottleneck=low_level_bottleneck,
            abn_block=abn_block,
            dropout=dropout,
        )

        self.return_full_size_mask = return_full_size_mask

    def forward(self, x):
        enc_features = self.encoder(x)

        # Decode mask
        mask, dsv = self.decoder(enc_features)

        if self.return_full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask, OUTPUT_MASK_32_KEY: dsv}

        return output


# resnet34-backbone models


def resnet34_fpncat128(input_channels=3, num_classes=1, dropout=0.0, pretrained=None):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


# seresnext50-backbone models


def seresnext50_fpncat128(input_channels=3, num_classes=1, dropout=0.0, pretrained=None):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)


# seresnext101-backbone models


def seresnext101_fpncat256(input_channels=3, num_classes=1, dropout=0.0, pretrained=None):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def seresnext101_fpnsum256(input_channels=3, num_classes=1, dropout=0.0, pretrained=None):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return FPNSumSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout)


def seresnext101_deeplab256(input_channels=3, num_classes=1, dropout=0.0):
    encoder = E.SEResNeXt101Encoder()
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return DeeplabV3SegmentationModel(encoder, num_classes=num_classes, high_level_bottleneck=256, dropout=dropout)


# efficientnet-backbone models


def efficientnetb4_fpncat128(input_channels=3, num_classes=1, dropout=0.0, pretrained=None):
    encoder = E.EfficientNetB4Encoder(abn_params={"activation": "swish"}, pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout)
