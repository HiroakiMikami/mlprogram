import numpy as np
import torch

from mlprogram.nn.functional import index_embeddings, position_embeddings


class TestPostionEmbeddings(object):
    def test_position_embeddings(self):
        b = 1
        E = 4
        indexes = np.array([[0, 1], [1, 0], [2, 2]])
        e_arr = indexes.astype(np.float) + b
        e_arr = e_arr.reshape(3, 2, 1)
        e_arr = np.broadcast_to(e_arr, (3, 2, E))
        divisors = np.array([1, 1, 10000**0.5, 10000**0.5])
        e_arr = np.sin(e_arr / divisors.reshape(1, 1, E))
        e_tensor = position_embeddings(torch.tensor(indexes), b, E)
        assert np.allclose(e_arr, e_tensor.numpy())

    def test_index_embeddings(self):
        b = 1
        E = 4
        indexes = np.array([[0], [1], [2]])
        e_arr = indexes.astype(np.float) + b
        e_arr = e_arr.reshape(3, 1, 1)
        e_arr = np.broadcast_to(e_arr, (3, 2, E))
        divisors = np.array([1, 1, 10000**0.5, 10000**0.5])
        e_arr = np.sin(e_arr / divisors.reshape(1, 1, E))
        tensor = torch.FloatTensor(3, 2, 4)
        e_tensor = index_embeddings(tensor, b)
        assert np.allclose(e_arr, e_tensor.numpy())
