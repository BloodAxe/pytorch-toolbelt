from torch import nn
from .common import DecoderModule
from ..fpn import FPNBottleneckBlock, UpsampleAdd, FPNPredictionBlock


class FPNDecoder(DecoderModule):
    def __init__(
        self,
        features,
        bottleneck=FPNBottleneckBlock,
        upsample_add_block=UpsampleAdd,
        prediction_block=FPNPredictionBlock,
        fpn_features=128,
        prediction_features=128,
        mode="bilinear",
        align_corners=False,
        upsample_scale=None,
    ):
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

        if isinstance(prediction_features, list) and len(prediction_features) != len(
            features
        ):
            raise ValueError()

        if not isinstance(fpn_features, list):
            fpn_features = [int(fpn_features)] * len(features)

        if not isinstance(prediction_features, list):
            prediction_features = [int(prediction_features)] * len(features)

        bottlenecks = [
            bottleneck(input_channels, output_channels)
            for input_channels, output_channels in zip(features, fpn_features)
        ]

        integrators = [
            upsample_add_block(
                output_channels,
                upsample_scale=upsample_scale,
                mode=mode,
                align_corners=align_corners,
            )
            for output_channels in fpn_features
        ]
        predictors = [
            prediction_block(input_channels, output_channels)
            for input_channels, output_channels in zip(
                fpn_features, prediction_features
            )
        ]

        self.bottlenecks = nn.ModuleList(bottlenecks)
        self.integrators = nn.ModuleList(integrators)
        self.predictors = nn.ModuleList(predictors)

        self.output_filters = prediction_features

    def forward(self, features):
        fpn_outputs = []
        prev_fpn = None
        for feature_map, bottleneck_module, upsample_add, output_module in zip(
            reversed(features),
            reversed(self.bottlenecks),
            reversed(self.integrators),
            reversed(self.predictors),
        ):
            curr_fpn = bottleneck_module(feature_map)
            curr_fpn = upsample_add(curr_fpn, prev_fpn)

            y = output_module(curr_fpn)
            prev_fpn = curr_fpn
            fpn_outputs.append(y)

        # Reverse list of fpn features to match with order of input features
        return list(reversed(fpn_outputs))
