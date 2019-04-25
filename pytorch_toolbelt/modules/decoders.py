import torch
from torch import nn

from .unet import UnetCentralBlock, UnetDecoderBlock
from .fpn import FPNFuse, FPNBottleneckBlock, FPNPredictionBlock

import torch.nn.functional as F


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)


class UNetDecoder(DecoderModule):
    def __init__(self, features, start_features: int, dilation_factors=[1, 1, 1, 1], **kwargs):
        super().__init__()
        decoder_features = start_features
        reversed_features = list(reversed(features))

        output_filters = [decoder_features]
        self.center = UnetCentralBlock(reversed_features[0], decoder_features)

        if dilation_factors is None:
            dilation_factors = [1] * len(reversed_features)

        blocks = []
        for block_index, encoder_features in enumerate(reversed_features):
            blocks.append(UnetDecoderBlock(output_filters[-1], encoder_features, decoder_features, dilation=dilation_factors[block_index]))
            output_filters.append(decoder_features)
            # print(block_index, decoder_features, encoder_features, decoder_features)
            decoder_features = decoder_features // 2

        self.blocks = nn.ModuleList(blocks)
        self.output_filters = output_filters

    def forward(self, features):
        reversed_features = list(reversed(features))
        decoder_outputs = [self.center(reversed_features[0])]

        for block_index, decoder_block, encoder_output in zip(range(len(self.blocks)), self.blocks, reversed_features):
            # print(block_index, decoder_outputs[-1].size(), encoder_output.size())
            decoder_outputs.append(decoder_block(decoder_outputs[-1], encoder_output))

        return decoder_outputs


class FPNDecoder(DecoderModule):
    def __init__(self, features,
                 prediction_block=FPNPredictionBlock,
                 bottleneck=FPNBottleneckBlock,
                 fpn_features=128,
                 prediction_features=128,
                 mode='bilinear',
                 align_corners=True,
                 upsample_scale=None):
        """

        :param features:
        :param prediction_block:
        :param bottleneck:
        :param fpn_features:
        :param prediction_features:
        :param mode:
        :param align_corners:
        :param upsample_scale: Scale factor for use during upsampling.
        By default it's None which infers targets size automatically.
        However, CoreML does not support this OP, so for CoreML-friendly models you may use fixed scale.
        """
        super().__init__()

        if isinstance(fpn_features, list) and len(fpn_features) != len(features):
            raise ValueError()

        if isinstance(prediction_features, list) and len(prediction_features) != len(features):
            raise ValueError()

        if not isinstance(fpn_features, list):
            fpn_features = [int(fpn_features)] * len(features)

        if not isinstance(prediction_features, list):
            prediction_features = [int(prediction_features)] * len(features)

        bottlenecks = [bottleneck(input_channels, output_channels) for input_channels, output_channels in zip(features, fpn_features)]
        predictors = [prediction_block(input_channels, output_channels) for input_channels, output_channels in zip(fpn_features, prediction_features)]

        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.predictors = nn.ModuleList(predictors)
        self.interpolation_mode = mode
        self.align_corners = align_corners
        self.output_filters = prediction_features
        self.upsample_scale = upsample_scale

    def _interpolate_add(self, bottleneck, prev_layer_bottleneck=None):
        """Compute bottleneck + Upsample(prev_layer_bottleneck)

        :param bottleneck:
        :param prev_layer_bottleneck:
        :return:
        """
        if prev_layer_bottleneck is not None:
            if self.upsample_scale is not None:
                prev_layer_bottleneck = F.interpolate(prev_layer_bottleneck,
                                                      scale_factor=self.upsample_scale,
                                                      mode=self.interpolation_mode,
                                                      align_corners=self.align_corners)
            else:
                prev_layer_bottleneck = F.interpolate(prev_layer_bottleneck,
                                                      size=(bottleneck.size(2), bottleneck.size(3)),
                                                      mode=self.interpolation_mode,
                                                      align_corners=self.align_corners)

            bottleneck = prev_layer_bottleneck + bottleneck

        return bottleneck

    def forward(self, features):
        fpn_outputs = []
        prev_fpn = None
        for feature_map, bottleneck_block, prediction_block in zip(reversed(features),
                                                                   reversed(self.bottlenecks),
                                                                   reversed(self.predictors)):
            curr_fpn = bottleneck_block(feature_map)
            curr_fpn = self._interpolate_add(curr_fpn, prev_fpn)

            y = prediction_block(curr_fpn)
            prev_fpn = curr_fpn
            fpn_outputs.append(y)

        # Reverse list of fpn features to match with order of input features
        return list(reversed(fpn_outputs))
