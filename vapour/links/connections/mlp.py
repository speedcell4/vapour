import itertools
from typing import List, Union

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable

from vapour.utils import compute_hidden_size

__all__ = [
    'MLP',
]


class MLP(Chain):
    def __init__(self, nb_layers: int, in_size: int, out_size: int, hidden_size: Union[int, List[int]] = None,
                 nonlinear=F.tanh, nobias: bool = False, initialW=None, initial_bias=None) -> None:
        super(MLP, self).__init__()

        if hidden_size is None:
            hidden_size = compute_hidden_size(in_size, out_size)
        hidden_layer_sizes = itertools.chain(itertools.repeat(hidden_size, nb_layers - 1), [out_size])

        if isinstance(hidden_size, list):
            assert len(hidden_size) == nb_layers - 1
            hidden_layer_sizes = hidden_size

        self.nb_layers = nb_layers
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.nonlinear = nonlinear

        with self.init_scope():
            for ix, layer_size in enumerate(hidden_layer_sizes):
                setattr(self, f'fc{ix}', L.Linear(
                    None, layer_size, nobias, initialW, initial_bias))

    def __call__(self, x: Variable) -> Variable:
        for ix in range(self.nb_layers - 1):
            x = self.nonlinear(getattr(self, f'fc{ix}')(x))
        return getattr(self, f'fc{self.nb_layers - 1}')(x)


if __name__ == '__main__':
    mlp = MLP(2, 4, 5)

    x = Variable(np.random.random((1, 4)).astype(np.float32))
    y = mlp(x)

    print(f'y :: {y.shape} => {y.data}')
