import torch

__all__ = ["clean_checkpoint"]


def clean_checkpoint(src_fname, dst_fname):
    """
    Removes optimizer, scheduler and criterion states from checkpoint
    :param src_fname: Source checkpoint filename
    :param dst_fname: Target checkpoint filename (can be same)
    """
    checkpoint = torch.load(src_fname, map_location="cpu")

    keys = ["criterion_state_dict", "optimizer_state_dict", "scheduler_state_dict"]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)
