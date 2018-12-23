import nnabla as nn
import nnabla.functions as F


def concatenate(*args, axis):
    if len(args) > 1:
        return F.concatenate(*args, axis=axis)
    else:
        return args[0]


def split(x, axis):
    if x.shape[axis] == 1:
        return [F.reshape(x, (*x.shape[:axis], *x.shape[axis + 1:]))]
    else:
        return F.split(x, axis=axis)


def embed_inverse(embed, n_inputs, n_features, base_axis=1):
    W = nn.parameter.get_parameter_or_create("embed/W", [n_inputs, n_features])
    W = F.transpose(W, axes=[1, 0])
    return F.affine(embed, W, base_axis=base_axis)
