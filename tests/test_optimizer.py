import collections
from typing import List, Dict

import torch.nn
from torch import nn
from torch.optim import SGD

from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from pytorch_toolbelt.utils import count_parameters


def count_parameters_in_param_groups(pg: List[Dict[str, List[torch.nn.Parameter]]]) -> Dict[str, int]:
    kv_iter = enumerate(pg)
    dict_iter = [(str(key), sum([p.numel() for p in value["params"]])) for key, value in kv_iter]
    return dict(dict_iter)


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
                ("neck", nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 64).requires_grad_(False))),
                ("decoder", nn.Sequential(conv_bn_relu(64, 32), conv_bn_relu(32, 1))),
            ]
        )
    )

    pg, defaults = build_optimizer_param_groups(
        model, learning_rate=1e-4, weight_decay=0, apply_weight_decay_on_bias=False, apply_weight_decay_on_norm=False
    )

    total_params = count_parameters(model)
    total_params_in_pg = count_parameters_in_param_groups(pg)
    assert SGD(pg, **defaults) is not None

    print(total_params)
    print(total_params_in_pg)
    assert len(pg) == 3
    assert sum(total_params_in_pg.values()) == total_params["trainable"]

    pg, defaults = build_optimizer_param_groups(
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

    print(total_params)
    print(total_params_in_pg)
    assert len(pg) == 10
    assert sum(total_params_in_pg.values()) == total_params["trainable"]
