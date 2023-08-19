from pprint import pprint

import pytest
import torch

import pytorch_toolbelt.modules.decoders as D
from pytorch_toolbelt.modules.interfaces import FeatureMapsSpecification
from pytorch_toolbelt.modules.upsample import UpsampleLayerType
from pytorch_toolbelt.utils.torch_utils import count_parameters, describe_outputs


@torch.no_grad()
@pytest.mark.parametrize(
    ("decoder_cls", "decoder_params"),
    [
        (D.FPNDecoder, {"out_channels": 128, "upsample_block": UpsampleLayerType.BILINEAR}),
        (D.FPNDecoder, {"out_channels": 128, "upsample_block": UpsampleLayerType.NEAREST}),
        (D.FPNDecoder, {"out_channels": 128, "upsample_block": UpsampleLayerType.DECONVOLUTION}),
        (D.BiFPNDecoder, {"out_channels": 128, "num_layers": 3}),
        (D.DeeplabV3PlusDecoder, {"out_channels": 128, "aspp_channels": 256}),
        (D.DeeplabV3Decoder, {"out_channels": 128, "aspp_channels": 256}),
        (D.CANDecoder, {"out_channels": 128}),
        #
        (D.UNetDecoder, {"out_channels": [128, 256, 384, 512], "upsample_block": UpsampleLayerType.DECONVOLUTION}),
        (D.UNetDecoder, {"out_channels": [128, 256, 384, 512], "upsample_block": UpsampleLayerType.NEAREST}),
        (D.UNetDecoder, {"out_channels": [128, 256, 384, 512], "upsample_block": UpsampleLayerType.BILINEAR}),
        (D.UNetDecoder, {"out_channels": [128, 256, 384, 512], "upsample_block": UpsampleLayerType.PIXEL_SHUFFLE}),
        (D.UNetDecoder, {"out_channels": [128, 256, 384, 512], "upsample_block": UpsampleLayerType.RESIDUAL_DECONV}),
    ],
)
def test_decoders(decoder_cls, decoder_params):
    input_spec = FeatureMapsSpecification(channels=(64, 128, 256, 512, 1024), strides=(4, 8, 16, 32, 64))
    input = input_spec.get_dummy_input()

    decoder = decoder_cls(input_spec, **decoder_params).eval()
    output = decoder(input)

    print()
    print(decoder.__class__.__name__)
    print(count_parameters(decoder, human_friendly=True))
    pprint(describe_outputs(output))

    torch.jit.trace(decoder, (input,))
