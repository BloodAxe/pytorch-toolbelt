import functools
import os.path
from typing import Optional, Callable, Any

from hydra import TaskFunction
from hydra._internal.utils import get_args_parser, _run_hydra
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "hydra_dpp_friendly_main",
    "nan_value",
    "register_albumentations_resolver",
    "probe_directory",
]


def nan_value():
    return float("nan")


def register_albumentations_resolver():
    def albumentations_resolver(key):
        return f"albumentations.{key}"

    OmegaConf.register_new_resolver("A", albumentations_resolver, use_cache=True)


def probe_directory(probe_dirs):
    for arg in probe_dirs:
        if os.path.exists(arg) and os.path.isdir(arg):
            return arg

    raise FileNotFoundError("None of the probe directories exists")


def hydra_dpp_friendly_main(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: the config path, a directory relative to the declaring python file.
    :param config_name: the name of the config (usually the file name without the .yaml extension)
    :param strict: (Deprecated) strict mode, will throw an error if command line overrides are not changing an
    existing key or if the code is accessing a non-existent key
    """

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args = get_args_parser()
                # Add local_rank to be able to use hydra with DDP
                args.add_argument("--local_rank", default=0, type=int)
                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                _run_hydra(
                    args_parser=args,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                )

        return decorated_main

    return main_decorator
