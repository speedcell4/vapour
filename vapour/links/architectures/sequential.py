from collections import OrderedDict

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Variable, Chain

from vapour.utils import ensure_tuple


class Sequential(Chain):
    def __init__(self, **chains):
        super(Sequential, self).__init__()
        self._layers = OrderedDict()

        with self.init_scope():
            for name, chain in chains.items():
                setattr(self, name, chain)
                self._layers[name] = chain

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (\n' + '\n'.join(
            f'  {layer},'
            for layer in self._layers.keys()
        ) + '\n)'

    def __call__(self, *args):
        for layer in self._layers.values():
            args = layer(*ensure_tuple(args))
        return args


if __name__ == '__main__':
    model = Sequential(
        fc0=L.Linear(None, 3),
        tanh1=F.tanh,
        fc1=L.Linear(None, 4),
        tanh2=F.tanh,
    )
    print(model)
    x = Variable(np.random.random((1, 3)).astype(np.float32))
    y = model.__call__(x)
    print(y.shape)
