import collections

from torch import nn

from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from pytorch_toolbelt.utils import count_parameters


def test_build_optimizer_param_groups():
    def conv_bn_relu(in_channels, out_channels):
        return nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    model = nn.Sequential(
        collections.OrderedDict(
            [
                ("encoder", nn.Sequential(conv_bn_relu(3, 32), conv_bn_relu(32, 64))),
                ("neck", nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 64))),
                ("decoder", nn.Sequential(conv_bn_relu(64, 32), conv_bn_relu(32, 1))),
            ]
        )
    )

    total_params = count_parameters(model)

    pg = build_optimizer_param_groups(
        model, learning_rate=1e-4, weight_decay=0, apply_weight_decay_on_bias=False, apply_weight_decay_on_norm=False
    )
    assert len(pg) == 1

    pg = build_optimizer_param_groups(
        model,
        learning_rate={"encoder": 1e-3, "neck": 1e-4, "decoder": 1e-5, "_default_": 0},
        weight_decay={
            "encoder.0": 1e-5,
            "encoder.1": 5e-5,
            "_default_": 1e-6,
        },
        apply_weight_decay_on_bias=False,
        apply_weight_decay_on_norm=False,
    )

    assert len(pg) == 7
