import torch
from pytorch_toolbelt.inference.functional import unpad_xyxy_bboxes


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
