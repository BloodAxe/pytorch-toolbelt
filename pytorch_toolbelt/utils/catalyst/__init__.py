from .criterions import *
from .callbacks import *
from .metrics import *
from .opl import *
from .visualization import *
from .utils import *
from .loss_adapter import *


def _register_modules(r):
    from pytorch_toolbelt.modules import encoders as e

    r.add_from_module(e, prefix="tbt.")


def _register_criterions(r):
    from pytorch_toolbelt import losses as l

    r.add_from_module(l, prefix="tbt.")


def _register_callbacks(r):
    from pytorch_toolbelt.utils.catalyst import callbacks as c

    r.add_from_module(c, prefix="tbt.")


def register_toolbelt_in_catalyst():
    """
    Register modules, losses & callbacks from pytorch-toolbelt in Catalyst
    """
    from catalyst.registry import MODULE, CRITERION

    MODULE.late_add(_register_modules)
    CRITERION.late_add(_register_criterions)
