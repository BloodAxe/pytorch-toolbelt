"""Utility functions to make your experiments reproducible

"""
import random
import torch


def set_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))


def get_rng_state() -> dict:
    return {
        'torch_rng': torch.get_rng_state(),
        'torch_rng_cuda': torch.cuda.get_rng_state_all(),
        'python_rng': random.getstate(),
    }


def set_rng_state(rng_state: dict):
    try:
        torch_rng = rng_state['torch_rng']
        torch.set_rng_state(torch_rng)
        print('Set torch rng state')
    except:
        pass

    try:
        torch_rng_cuda = rng_state['torch_rng_cuda']
        torch.cuda.set_rng_state(torch_rng_cuda)
        print('Set torch rng cuda state')
    except:
        pass

    try:
        python_rng = rng_state['python_rng']
        random.setstate(python_rng)
        print('Set python rng state')
    except:
        pass


def get_random_name() -> str:
    from . import namesgenerator as ng
    return ng.get_random_name()
