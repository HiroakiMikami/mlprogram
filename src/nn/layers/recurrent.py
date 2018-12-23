import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from .utils import split, concatenate
from .initializers import orthogonal


def lstm(x,
         mask,
         state_size,
         w_init=None,
         inner_w_init=None,
         forget_bias_init=I.ConstantInitializer(1),
         b_init=I.ConstantInitializer(0),
         initial_state=None,
         dropout=0,
         train=True,
         rng=np.random):
    """
    x: (batch_size, length, input_size)
    mask: (batch_size, length)
    """
    batch_size, length, input_size = x.shape

    if w_init is None:
        w_init = I.UniformInitializer(
            I.calc_uniform_lim_glorot(input_size, state_size))
    if inner_w_init is None:
        inner_w_init = orthogonal

    retain_prob = 1.0 - dropout
    z_w = nn.Variable((batch_size, 4, input_size), need_grad=False)
    z_w.d = 1
    z_u = nn.Variable((batch_size, 4, state_size), need_grad=False)
    z_u.d = 1

    if dropout > 0:
        if train:
            z_w = F.dropout(z_w, p=retain_prob)
            z_u = F.dropout(z_u, p=retain_prob)
        z_w *= retain_prob
        z_u *= retain_prob

    z_w = F.reshape(z_w, (batch_size, 4, 1, input_size))
    z_w = F.broadcast(z_w, (batch_size, 4, length, input_size))
    z_w = F.split(z_w, axis=1)
    z_u = F.split(z_u, axis=1)
    xi = z_w[0] * x
    xf = z_w[1] * x
    xc = z_w[2] * x
    xo = z_w[3] * x

    with nn.parameter_scope("lstm"):
        # (batch_size, length, state_size)
        xi = PF.affine(
            xi,
            state_size,
            base_axis=2,
            w_init=w_init,
            b_init=b_init,
            name="Wi")
        xf = PF.affine(
            xf,
            state_size,
            base_axis=2,
            w_init=w_init,
            b_init=forget_bias_init,
            name="Wf")
        xc = PF.affine(
            xc,
            state_size,
            base_axis=2,
            w_init=w_init,
            b_init=b_init,
            name="Wc")
        xo = PF.affine(
            xo,
            state_size,
            base_axis=2,
            w_init=w_init,
            b_init=b_init,
            name="Wo")

    if initial_state is None:
        h = nn.Variable((batch_size, state_size), need_grad=False)
        h.data.zero()
    else:
        h = initial_state
    c = nn.Variable((batch_size, state_size), need_grad=False)
    c.data.zero()

    # (batch_size, state_size)
    xi = split(xi, axis=1)
    xf = split(xf, axis=1)
    xc = split(xc, axis=1)
    xo = split(xo, axis=1)
    mask = F.reshape(mask, [batch_size, length, 1])  # (batch_size, length, 1)
    mask = F.broadcast(mask, [batch_size, length, state_size])
    # (batch_size, state_size)
    mask = split(mask, axis=1)

    hs = []
    cs = []
    with nn.parameter_scope("lstm"):
        for i, f, c2, o, m in zip(xi, xf, xc, xo, mask):
            i_t = PF.affine(
                z_u[0] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Ui")
            i_t = F.sigmoid(i + i_t)
            f_t = PF.affine(
                z_u[1] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uf")
            f_t = F.sigmoid(f + f_t)
            c_t = PF.affine(
                z_u[2] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uc")
            c_t = f_t * c + i_t * F.tanh(c2 + c_t)
            o_t = PF.affine(
                z_u[3] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uo")
            o_t = F.sigmoid(o + o_t)
            h_t = o_t * F.tanh(c_t)

            h_t = (1 - m) * h + m * h_t
            c_t = (1 - m) * c + m * c_t
            h = h_t
            c = c_t
            h_t = F.reshape(h_t, (batch_size, 1, state_size), inplace=False)
            c_t = F.reshape(c_t, (batch_size, 1, state_size), inplace=False)
            hs.append(h_t)
            cs.append(c_t)
    return concatenate(*hs, axis=1), concatenate(*cs, axis=1)


def bilstm(x,
           mask,
           state_size,
           w_init=None,
           inner_w_init=None,
           forget_bias_init=I.ConstantInitializer(1),
           b_init=I.ConstantInitializer(0),
           initial_state=None,
           dropout=0,
           train=True,
           rng=np.random):
    rx = F.flip(x, axes=[1])  # reverse
    rmask = F.flip(mask, axes=[1])  # reverse
    with nn.parameter_scope("forward"):
        hs, _ = lstm(x, mask, state_size, w_init, inner_w_init,
                     forget_bias_init, b_init, initial_state, dropout, train,
                     rng)
    with nn.parameter_scope("backward"):
        rhs, _ = lstm(rx, rmask, state_size, w_init, inner_w_init,
                      forget_bias_init, b_init, initial_state, dropout, train,
                      rng)
    hs2 = F.flip(rhs, axes=[1])  # reverse
    return concatenate(hs, hs2, axis=2)  # (batch_size, length, 2 * state_size)
