from typing import Any, Dict

__all__ = ["maybe_eval", "without"]


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
    Returns copy of dictionary without given key.

    Args:
        dictionary: Input dictionary
        key: Key to remove

    Returns:
        Always returns new dictionary even without given key
    """
    new_d = dictionary.copy()
    if key in dictionary:
        new_d.pop(key)
    return new_d
