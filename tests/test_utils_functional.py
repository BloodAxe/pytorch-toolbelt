import torch
from torch import nn
import torch.nn.functional as F
from pytorch_toolbelt.inference.functional import unpad_xyxy_bboxes
from pytorch_toolbelt.modules.encoders import make_n_channel_input


def test_unpad_xyxy_bboxes():
    bboxes1 = torch.rand((1, 32, 4))
    bboxes2 = torch.rand((1, 32, 4, 20))

    pad = [2, 3, 4, 5]
    bboxes1_unpad = unpad_xyxy_bboxes(bboxes1, pad, dim=-1)
    assert bboxes1_unpad.size(0) == 1
    assert bboxes1_unpad.size(1) == 32
    assert bboxes1_unpad.size(2) == 4

    bboxes2_unpad = unpad_xyxy_bboxes(bboxes2, pad, dim=2)
    assert bboxes2_unpad.size(0) == 1
    assert bboxes2_unpad.size(1) == 32
    assert bboxes2_unpad.size(2) == 4
    assert bboxes2_unpad.size(3) == 20


def test_make_n_channel_input():
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    conv6 = make_n_channel_input(conv, in_channels=6)
    assert conv6.weight.size(0) == conv.weight.size(0)
    assert conv6.weight.size(1) == 6
    assert conv6.weight.size(2) == conv.weight.size(2)
    assert conv6.weight.size(3) == conv.weight.size(3)

    conv5 = make_n_channel_input(conv, in_channels=5)
    assert conv5.weight.size(0) == conv.weight.size(0)
    assert conv5.weight.size(1) == 5
    assert conv5.weight.size(2) == conv.weight.size(2)
    assert conv5.weight.size(3) == conv.weight.size(3)
