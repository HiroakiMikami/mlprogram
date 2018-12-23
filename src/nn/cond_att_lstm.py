import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from .layers.utils import split, concatenate
from .layers.initializers import orthogonal


def cond_att_lstm(x,
                  parent_index,
                  mask,
                  context,
                  context_mask,
                  state_size,
                  att_hidden_size,
                  initial_state=None,
                  initial_cell=None,
                  hist=None,
                  dropout=0,
                  train=True,
                  w_init=None,
                  inner_w_init=None,
                  b_init=I.ConstantInitializer(0),
                  forget_bias_init=I.ConstantInitializer(1)):
    """
    x: (batch_size, length, input_size)
    parent_index: (batch_size, length)
    mask: (batch_size, length)
    context: (batch_size, context_length, context_size)
    context_mask: (batch_size, context_length)
    hist: (batch_size, l, state_size)
    """
    batch_size, length, input_size = x.shape
    _, context_length, context_size = context.shape

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

    with nn.parameter_scope("cond_att_lstm"):
        # (batch_size, length, state_size)
        with nn.parameter_scope("lstm"):
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

        with nn.parameter_scope("context"):
            # context_att_trans: (batch_size, context_size, att_hidden_size)
            context_att_trans = PF.affine(
                context,
                att_hidden_size,
                base_axis=2,
                w_init=w_init,
                b_init=b_init,
                name="layer1_c")

    if initial_state is None:
        h = nn.Variable((batch_size, state_size), need_grad=False)
        h.data.zero()
    else:
        h = initial_state

    if initial_cell is None:
        c = nn.Variable((batch_size, state_size), need_grad=False)
        c.data.zero()
    else:
        c = initial_cell

    if hist is None:
        hist = nn.Variable((batch_size, 1, state_size), need_grad=False)
        hist.data.zero()

    # (batch_size, state_size)
    xi = split(xi, axis=1)
    xf = split(xf, axis=1)
    xc = split(xc, axis=1)
    xo = split(xo, axis=1)
    mask = F.reshape(mask, [batch_size, length, 1])  # (batch_size, length, 1)
    mask = F.broadcast(mask, [batch_size, length, state_size])
    # (batch_size, state_size)
    mask = split(mask, axis=1)
    # (batch_size, max_action_length)
    parent_index = parent_index + 1  # index == 0 means that parent is root
    # (batch_size)
    parent_index = split(parent_index, axis=1)

    hs = []
    cs = []
    ctx = []

    for i, f, c2, o, m, p in zip(xi, xf, xc, xo, mask, parent_index):
        h_num = hist.shape[1]
        with nn.parameter_scope("context"):
            h_att_trans = PF.affine(
                h,
                att_hidden_size,
                with_bias=False,
                w_init=w_init,
                name="layer1_h")  # (batch_size, att_hidden_size)
            h_att_trans = F.reshape(h_att_trans,
                                    (batch_size, 1, att_hidden_size))
            h_att_trans = F.broadcast(
                h_att_trans, (batch_size, context_length, att_hidden_size))
            att_hidden = F.tanh(context_att_trans + h_att_trans)
            att_raw = PF.affine(
                att_hidden, 1, base_axis=2, w_init=w_init,
                b_init=b_init)  # (batch_size, context_length, 1)
            att_raw = F.reshape(att_raw, (batch_size, context_length))
            ctx_att = F.exp(att_raw - F.max(att_raw, axis=1, keepdims=True))
            ctx_att = ctx_att * context_mask
            ctx_att = ctx_att / F.sum(ctx_att, axis=1, keepdims=True)
            ctx_att = F.reshape(ctx_att, (batch_size, context_length, 1))
            ctx_att = F.broadcast(ctx_att,
                                  (batch_size, context_length, context_size))
            ctx_vec = F.sum(
                context * ctx_att, axis=1)  # (batch_size, context_size)

        # parent_history
        p = F.reshape(p, (batch_size, 1))
        p = F.one_hot(p, (h_num, ))
        p = F.reshape(p, (batch_size, 1, h_num))
        par_h = F.batch_matmul(p, hist)  # [batch_size, 1, state_size]
        par_h = F.reshape(par_h, (batch_size, state_size))

        with nn.parameter_scope("lstm"):
            i_t = PF.affine(
                z_u[0] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Ui")
            i_t += PF.affine(
                ctx_vec,
                state_size,
                w_init=inner_w_init(context_size, state_size),
                with_bias=False,
                name="Ci")
            i_t += PF.affine(
                par_h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Pi")
            i_t = F.sigmoid(i + i_t)
            f_t = PF.affine(
                z_u[1] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uf")
            f_t += PF.affine(
                ctx_vec,
                state_size,
                w_init=inner_w_init(context_size, state_size),
                with_bias=False,
                name="Cf")
            f_t += PF.affine(
                par_h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Pf")
            f_t = F.sigmoid(f + f_t)
            c_t = PF.affine(
                z_u[2] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uc")
            c_t += PF.affine(
                ctx_vec,
                state_size,
                w_init=inner_w_init(context_size, state_size),
                with_bias=False,
                name="Cc")
            c_t += PF.affine(
                par_h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Pc")
            c_t = f_t * c + i_t * F.tanh(c2 + c_t)
            o_t = PF.affine(
                z_u[3] * h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Uo")
            o_t += PF.affine(
                ctx_vec,
                state_size,
                w_init=inner_w_init(context_size, state_size),
                with_bias=False,
                name="Co")
            o_t += PF.affine(
                par_h,
                state_size,
                w_init=inner_w_init(state_size, state_size),
                with_bias=False,
                name="Po")
            o_t = F.sigmoid(o + o_t)
            h_t = o_t * F.tanh(c_t)

            h_t = (1 - m) * h + m * h_t
            c_t = (1 - m) * c + m * c_t
            h = h_t
            c = c_t
            h_t = F.reshape(h_t, (batch_size, 1, state_size), inplace=False)
            c_t = F.reshape(c_t, (batch_size, 1, state_size), inplace=False)
            ctx_vec = F.reshape(
                ctx_vec, (batch_size, 1, context_size), inplace=False)
            hs.append(h_t)
            cs.append(c_t)
            ctx.append(ctx_vec)

            hist = F.concatenate(
                hist, h_t, axis=1)  # (batch_size, h_num + 1, state_size)

    return concatenate(
        *hs, axis=1), concatenate(
            *cs, axis=1), concatenate(
                *ctx, axis=1), hist
