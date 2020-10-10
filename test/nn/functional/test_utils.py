import torch

from mlprogram.nn.functional import nel_to_lne, lne_to_nel


class TestEmbeddings(object):
    def test_nel_to_lne(self):
        assert (4, 2, 3) == nel_to_lne(torch.Tensor(2, 3, 4)).shape

    def test_lne_to_nel(self):
        assert (3, 4, 2) == lne_to_nel(torch.Tensor(2, 3, 4)).shape
