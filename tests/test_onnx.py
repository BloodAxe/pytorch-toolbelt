import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.backbone.inceptionv4 import inceptionv4
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters
from pytorch_toolbelt.modules.encoders import timm

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        # [timm.B0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        # [timm.MixNetXLEncoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SKResNet18Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SWSLResNeXt101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmResnet200D, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.HRNetW18Encoder, {"pretrained": False}],
        [timm.DPN68Encoder, {"pretrained": False}],
        [timm.NFNetF0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmRes2Next50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
    ],
)
@skip_if_no_cuda
def test_onnx_export(encoder, encoder_params):
    import onnx

    model = encoder(**encoder_params).eval()

    print(model.__class__.__name__, count_parameters(model))
    print(model.strides)
    print(model.channels)
    dummy_input = torch.rand((1, 3, 256, 256))
    dummy_input = maybe_cuda(dummy_input)
    model = maybe_cuda(model)

    input_names = ["image"]
    output_names = [f"feature_map_{i}" for i in range(len(model.channels))]

    torch.onnx.export(model, dummy_input, "tmp.onnx", verbose=True, input_names=input_names, output_names=output_names)
    model = onnx.load("tmp.onnx")
    onnx.checker.check_model(model)


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [timm.B0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.MixNetXLEncoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SKResNet18Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SWSLResNeXt101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmResnet200D, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.HRNetW18Encoder, {"pretrained": False}],
        [timm.DPN68Encoder, {"pretrained": False}],
        [timm.NFNetF0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmRes2Next50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
    ],
)
@skip_if_no_cuda
def test_jit_trace(encoder, encoder_params):
    model = encoder(**encoder_params).eval()

    print(model.__class__.__name__, count_parameters(model))
    print(model.strides)
    print(model.channels)
    dummy_input = torch.rand((1, 3, 256, 256))
    dummy_input = maybe_cuda(dummy_input)
    model = maybe_cuda(model)

    model = torch.jit.trace(model, dummy_input, check_trace=True)
