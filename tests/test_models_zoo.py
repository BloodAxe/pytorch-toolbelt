import pytest
import torch

from pytorch_toolbelt.zoo import resnet34_unet32_s2, resnet34_unet64_s4, hrnet34_unet64

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@skip_if_no_cuda
@torch.no_grad()
@pytest.mark.parametrize("model_cls", [resnet34_unet32_s2, resnet34_unet64_s4, hrnet34_unet64])
def test_segmentation_models(model_cls):
    num_classes = 7
    net = model_cls(num_classes=num_classes).cuda().eval()
    input = torch.randn((4, 3, 512, 512)).cuda()

    with torch.cuda.amp.autocast(True):
        output = net(input)

    assert output.size(0) == input.size(0)
    assert output.size(1) == num_classes
    assert output.size(2) == input.size(2)
    assert output.size(3) == input.size(3)
