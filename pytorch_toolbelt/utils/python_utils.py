import numbers
from typing import Any, Dict, Tuple, Union, Iterable
from pytorch_toolbelt.utils.support import pytorch_toolbelt_deprecated

__all__ = ["maybe_eval", "without", "load_yaml", "as_tuple_of_two"]


def maybe_eval(x: str) -> Any:
    if isinstance(x, str):
        if x.startswith("$"):
            return eval(x[1:])
        return x
    elif isinstance(x, list):
        return list(map(maybe_eval, x))
    else:
        return x


def without(dictionary: Dict, key: str) -> Dict:
    """
    Return copy of dictionary without given key.

    Args:
        dictionary: Input dictionary
        key: Key to remove

    Returns:
        Always returns new dictionary even without given key
    """
    if isinstance(key, str):
        key = {key}
    return dict((k, v) for (k, v) in dictionary.items() if k not in key)


@pytorch_toolbelt_deprecated("This method is deprecated. Please use OmegaConf")
def load_yaml(stream: Any):
    """
    Parse the first YAML document in a stream and produce the corresponding Python object.
    This function support parsing float values like `1e-4`.

    Implementation credit: https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number

    Args:
        stream:
    """
    import yaml
    import re

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    return yaml.load(stream, Loader=loader)


def as_tuple_of_two(
    value: Union[numbers.Number, Tuple[numbers.Number, numbers.Number]]
) -> Tuple[numbers.Number, numbers.Number]:
    """
    Takes single number of tuple of two numbers and always returns a tuple of two numbers
    Args:
        value:

    Returns:
        512     - (512,512)
        256,257 - (256, 257)
    """
    if isinstance(value, Iterable):
        a, b = value
        return a, b
    if isinstance(value, numbers.Number):
        return value, value
    raise RuntimeError(f"Unsupported input value {value}")
