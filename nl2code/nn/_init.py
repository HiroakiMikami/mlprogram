import numpy as np
import torch


def orthogonal_(tensor: torch.Tensor, gain=1.1):
    input_size = tensor.shape[0]
    output_size = tensor.shape[1]
    a = np.random.normal(0.0, 1.0, (input_size, output_size))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (input_size, output_size) else v
    q = q.reshape((input_size, output_size))
    q = gain * q[:input_size, :output_size]
    q = torch.Tensor(q)
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
    return tensor
