from .common import GenericTimmEncoder
from ..common import EncoderModule, _take, make_n_channel_input
from ...activations import ACT_RELU, get_activation_block
from pytorch_toolbelt.modules.interfaces import FeatureMapsSpecification

__all__ = ["HRNetW18Encoder", "HRNetW32Encoder", "HRNetW48Encoder", "TimmHRNetW18SmallV2Encoder"]


class HRNetTimmEncoder(GenericTimmEncoder):
    def __init__(self, encoder, first_conv_stride_one, layers):
        if first_conv_stride_one:
            encoder.conv1.stride = (1, 1)

        super().__init__(encoder, layers)
        if first_conv_stride_one:
            self.output_spec = FeatureMapsSpecification(
                channels=self.output_spec.channels, strides=tuple([s // 2 for s in self.output_spec.strides])
            )

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW18Encoder(HRNetTimmEncoder):
    def __init__(
        self, pretrained=True, use_incre_features: bool = True, layers=None, first_conv_stride_one: bool = False
    ):
        from timm.models import hrnet

        encoder = hrnet.hrnet_w18(
            pretrained=pretrained,
            feature_location="incre" if use_incre_features else "",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        super().__init__(encoder, first_conv_stride_one=first_conv_stride_one, layers=layers)


class HRNetW32Encoder(HRNetTimmEncoder):
    def __init__(
        elf, pretrained=True, use_incre_features: bool = True, layers=None, first_conv_stride_one: bool = False
    ):
        from timm.models import hrnet

        encoder = hrnet.hrnet_w32(
            pretrained=pretrained,
            feature_location="incre" if use_incre_features else "",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        super().__init__(encoder, first_conv_stride_one=first_conv_stride_one, layers=layers)


class HRNetW48Encoder(HRNetTimmEncoder):
    def __init__(
        elf, pretrained=True, use_incre_features: bool = True, layers=None, first_conv_stride_one: bool = False
    ):
        from timm.models import hrnet

        encoder = hrnet.hrnet_w48(
            pretrained=pretrained,
            feature_location="incre" if use_incre_features else "",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        super().__init__(encoder, first_conv_stride_one=first_conv_stride_one, layers=layers)


class TimmHRNetW18SmallV2Encoder(HRNetTimmEncoder):
    def __init__(
        self, elf, pretrained=True, use_incre_features: bool = True, layers=None, first_conv_stride_one: bool = False
    ):
        from timm.models import hrnet

        encoder = hrnet.hrnet_w18_small_v2(
            pretrained=pretrained,
            feature_location="incre" if use_incre_features else "",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        super().__init__(encoder, first_conv_stride_one=first_conv_stride_one, layers=layers)
