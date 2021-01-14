"""Utility functions to make your experiments reproducible

"""
import random
import warnings

import torch

__all__ = ["set_manual_seed", "get_rng_state", "set_rng_state", "get_random_name"]


def set_manual_seed(seed):
    """Set random seed for Python and PyTorch random generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))


def get_rng_state() -> dict:
    return {
        "torch_rng": torch.get_rng_state(),
        "torch_rng_cuda": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
    }


def set_rng_state(rng_state: dict):
    try:
        torch_rng = rng_state["torch_rng"]
        torch.set_rng_state(torch_rng)
        print("Set torch rng state")
    except ValueError as e:
        warnings.warn(e)

    try:
        torch_rng_cuda = rng_state["torch_rng_cuda"]
        torch.cuda.set_rng_state(torch_rng_cuda)
        print("Set torch rng cuda state")
    except ValueError as e:
        warnings.warn(e)

    try:
        python_rng = rng_state["python_rng"]
        random.setstate(python_rng)
        print("Set python rng state")
    except ValueError as e:
        warnings.warn(e)


def get_random_name() -> str:
    from . import namesgenerator as ng

    return ng.get_random_name()
