from torch import nn

__all__ = ["instantiate_normalization_block", "NORM_BATCH", "NORM_INSTANCE", "NORM_GROUP"]


NORM_BATCH = "batch_norm"
NORM_INSTANCE = "instance_norm"
NORM_GROUP = "group_norm"


def instantiate_normalization_block(normalization: str, in_channels: int, **kwargs):
    if normalization in ("bn", "batch", "batch_norm", "batchnorm"):
        return nn.BatchNorm2d(num_features=in_channels)

    if normalization in ("gn", "group", "group_norm", "groupnorm"):
        return nn.GroupNorm(num_channels=in_channels, **kwargs)

    if normalization in ("in", "instance", "instance_norm", "instancenorm"):
        return nn.InstanceNorm2d(num_features=in_channels, **kwargs)

    raise KeyError(f"Unknown normalization type '{normalization}'")
