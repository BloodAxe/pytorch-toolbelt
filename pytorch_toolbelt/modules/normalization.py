from torch import nn

__all__ = ["instantiate_normalization_block", "NORM_BATCH", "NORM_INSTANCE", "NORM_GROUP"]


NORM_BATCH = "batch_norm"
NORM_INSTANCE = "instance_norm"
NORM_GROUP = "group_norm"


def instantiate_normalization_block(normalization: str, in_channels: int, **kwargs):
    if normalization in ("bn", "batch", "batch2d", "batch_norm", "batch_norm_2d", "batchnorm", "batchnorm2d"):
        return nn.BatchNorm2d(num_features=in_channels)

    if normalization in ("bn3d", "batch3d", "batch_norm3d", "batch_norm_3d", "batchnorm3d"):
        return nn.BatchNorm3d(num_features=in_channels)

    if normalization in ("gn", "group", "group_norm", "groupnorm"):
        return nn.GroupNorm(num_channels=in_channels, **kwargs)

    if normalization in (
        "in",
        "instance",
        "instance2d",
        "instance_norm",
        "instancenorm",
        "instance_norm_2d",
        "instancenorm2d",
    ):
        return nn.InstanceNorm2d(num_features=in_channels, **kwargs)

    if normalization in ("in3d", "instance3d", "instance_norm_3d", "instancenorm3d"):
        return nn.InstanceNorm3d(num_features=in_channels, **kwargs)

    raise KeyError(f"Unknown normalization type '{normalization}'")
