import unittest
import torch

from mlprogram.nn.functional import nel_to_lne, lne_to_nel


class TestEmbeddings(unittest.TestCase):
    def test_nel_to_lne(self):
        self.assertEqual(
            (4, 2, 3),
            nel_to_lne(torch.Tensor(2, 3, 4)).shape
        )

    def test_lne_to_nel(self):
        self.assertEqual(
            (3, 4, 2),
            lne_to_nel(torch.Tensor(2, 3, 4)).shape
        )


if __name__ == "__main__":
    unittest.main()
