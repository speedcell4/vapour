import numpy as np

__all__ = [
    'ensure_tuple',
    'clip',
    'compute_hidden_size',
]


def ensure_tuple(item):
    if isinstance(item, (set, list, tuple)):
        return tuple(item)
    return item,


def clip(value, min_value=None, max_value=None):
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def compute_hidden_size(in_size: int, out_size: int, min_value: int = None, max_value: int = None) -> int:
    vec_size = int(np.maximum(in_size, out_size) + np.ceil(np.sqrt(in_size * out_size)))
    return clip(vec_size, min_value, max_value)
