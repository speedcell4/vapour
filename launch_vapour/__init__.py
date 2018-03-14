import warnings

import chainer


def launch_vapour(device: int = -1, random_seed: int = 333) -> None:
    import numpy
    import random

    random.seed(random_seed)
    numpy.random.seed(random_seed)

    if device >= 0:
        try:
            import cupy

            cupy.cuda.runtime.setDevice(device)
            cupy.random.seed(random_seed)
        except Exception as e:
            warnings.warn(f'launch_vapour :: {e}')
    print("random seed: {}".format(random_seed), file=chainer.config.stdlog)
