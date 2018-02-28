import numpy as np

__all__ = [
    'ensure_tuple',
    'compute_hidden_size',
]


def ensure_tuple(item):
    if isinstance(item, (set, list, tuple)):
        return tuple(item)
    return item,


def compute_hidden_size(in_size: int, out_size: int) -> int:
    return int(np.maximum(in_size, out_size) + np.ceil(np.sqrt(in_size * out_size)))
