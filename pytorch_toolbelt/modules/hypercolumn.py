"""Implementation of hypercolumn module from "Hypercolumns for Object Segmentation and Fine-grained Localization"

Original paper: https://arxiv.org/abs/1411.5752
"""

from .fpn import FPNFuse

__all__ = ["HyperColumn"]

HyperColumn = FPNFuse
