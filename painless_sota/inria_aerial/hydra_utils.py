import copy
import functools
import os.path
import pickle
import warnings
from pathlib import Path
from typing import Optional, Callable, Any, List

from hydra import TaskFunction
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _flush_loggers, configure_log
from omegaconf import DictConfig, OmegaConf, read_write, open_dict

_UNSPECIFIED_: Any = object()

__all__ = [
    "hydra_dpp_friendly_main",
    "register_albumentations_resolver",
    "probe_directory",
]


def register_albumentations_resolver():
    def albumentations_resolver(key):
        return f"albumentations.{key}"

    if not OmegaConf.has_resolver("A"):
        OmegaConf.register_new_resolver("A", albumentations_resolver, use_cache=True)


def probe_directory(probe_dirs):
    for arg in probe_dirs:
        arg = os.path.expanduser(arg)
        if os.path.exists(arg) and os.path.isdir(arg):
            return arg

    raise FileNotFoundError("None of the probe directories exists")


def _get_rerun_conf(file_path: str, overrides: List[str]) -> DictConfig:
    msg = "Experimental rerun CLI option, other command line args are ignored."
    warnings.warn(msg, UserWarning)
    file = Path(file_path)
    if not file.exists():
        raise ValueError(f"File {file} does not exist!")

    if len(overrides) > 0:
        msg = "Config overrides are not supported as of now."
        warnings.warn(msg, UserWarning)

    with open(str(file), "rb") as input:
        config = pickle.load(input)  # nosec
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    HydraConfig.instance().set_config(config)
    task_cfg = copy.deepcopy(config)
    with read_write(task_cfg):
        with open_dict(task_cfg):
            del task_cfg["hydra"]
    assert isinstance(task_cfg, DictConfig)
    return task_cfg


def hydra_dpp_friendly_main(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
    strict: Optional[bool] = None,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: the config path, a directory relative to the declaring python file.
    :param config_name: the name of the config (usually the file name without the .yaml extension)
    :param strict: (Deprecated) strict mode, will throw an error if command line overrides are not changing an
    existing key or if the code is accessing a non existent key
    """
    from hydra import version
    from hydra._internal.deprecation_warning import deprecation_warning
    from hydra._internal.utils import _run_hydra, get_args_parser
    from textwrap import dedent

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/next/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                # Add local_rank to be able to use hydra with DDP
                args_parser.add_argument("--local_rank", default=0, type=int)
                args = args_parser.parse_args()
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )

        return decorated_main

    return main_decorator
