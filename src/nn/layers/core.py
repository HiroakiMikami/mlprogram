import numpy as np

import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


def dense(x,
          output_dim,
          base_axis=1,
          w_init=None,
          b_init=I.ConstantInitializer(0),
          activation=F.tanh):
    if w_init is None:
        w_init = I.UniformInitializer(
            I.calc_uniform_lim_glorot(np.prod(x.shape[1:]), output_dim))
    return activation(
        PF.affine(
            x, output_dim, base_axis=base_axis, w_init=w_init, b_init=b_init))
