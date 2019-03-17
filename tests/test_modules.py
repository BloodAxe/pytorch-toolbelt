import pytest
import torch
import pytoch_toolbelt

@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [SqueezenetEncoder, {'layers': [0, 1, 2, 3]}],
    [MobilenetV2Encoder, {'layers': [0, 1, 2, 3, 4, 5, 6, 7]}],
    [Resnet18Encoder, {}],
    [Resnet34Encoder, {}],
    [Resnet50Encoder, {}],
    [Resnet101Encoder, {}],
    [Resnet152Encoder, {}],
    [SEResNeXt50Encoder, {}],
    [SEResnet101Encoder, {}],
])
def test_encoders(encoder, encoder_params):
    net = encoder(**encoder_params).eval().cuda()
    print(net.__class__.__name__, count_parameters(net))
    with torch.no_grad():
        x = torch.rand((4, 3, 512, 512)).cuda()
        y = net(x)
        for yi in y:
            print(yi.size())
