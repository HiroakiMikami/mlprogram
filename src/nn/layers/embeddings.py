import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


def embedding(x, input_dim, output_dim, init=None, mask_zero=False):
    if init is None:
        init = I.UniformInitializer((-0.1, 0.1))
    initialized = "embed/W" in nn.get_parameters()
    result = PF.embed(x, input_dim, output_dim)
    if not initialized:
        nn.get_parameters()["embed/W"].d = init(
            nn.get_parameters()["embed/W"].shape)

    if mask_zero:
        return result, 1 - F.equal_scalar(x, 0)
    else:
        return result
