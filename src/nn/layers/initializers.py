import numpy as np


def orthogonal(input_size, output_size, scale=1.1):
    a = np.random.normal(0.0, 1.0, (input_size, output_size))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (input_size, output_size) else v
    q = q.reshape((input_size, output_size))
    return scale * q[:input_size, :output_size]
