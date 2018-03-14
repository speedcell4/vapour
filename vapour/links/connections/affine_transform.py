import chainer.functions as F
import chainer.links as L
from chainer import Variable, Chain
import numpy as np


class AffineTransform(Chain):
    def __init__(self, *in_sizes: int, out_size: int,
                 nonlinear=F.tanh, nobias: bool = False, initialW=None, initial_bias=None) -> None:
        super(AffineTransform, self).__init__()

        self.in_size = sum(in_sizes)
        self.out_size = out_size
        self.nonlinear = nonlinear
        self.nobias = nobias

        self.fc = L.Linear(self.in_size, self.out_size, nobias, initialW, initial_bias)

    def __call__(self, *xs: Variable, axis: int = 1) -> Variable:
        return self.fc(F.concat(xs, axis=axis))


if __name__ == '__main__':
    affine_transform = AffineTransform(3, 4, 5, out_size=6)
    x1 = Variable(np.random.random((1, 3)).astype(np.float32))
    x2 = Variable(np.random.random((1, 4)).astype(np.float32))
    x3 = Variable(np.random.random((1, 5)).astype(np.float32))

    y = affine_transform(x1, x2, x3)

    print(f'y :: {y.shape} => {y.data}')
