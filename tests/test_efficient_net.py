from pytorch_toolbelt.modules.backbone.efficient_net import efficient_net_b0
from pytorch_toolbelt.modules.backbone.mobilenet import MobileNetV2
from pytorch_toolbelt.utils.torch_utils import count_parameters


def test_efficient_net():
    model = efficient_net_b0(num_classes=10)
    print(count_parameters(model))

    model = MobileNetV2(n_class=10)
    print(count_parameters(model))
