import torch

__all__ = ["clean_checkpoint", "report_checkpoint"]

from catalyst.dl import Callback, CallbackOrder, RunnerState


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


def report_checkpoint(checkpoint):
    """
    Prints checkpoint metrics & epoch
    :param checkpoint:
    """
    print("Epoch          :", checkpoint["epoch"])

    skip_fields = [
        "_base/lr",
        "_base/momentum",
        "_timers/data_time",
        "_timers/model_time",
        "_timers/batch_time",
        "_timers/_fps",
    ]
    print(
        "Metrics (Train):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["train"].items() if k not in skip_fields]
    )
    print(
        "Metrics (Valid):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["valid"].items() if k not in skip_fields]
    )
