import unittest
from src.nn.layers.utils import embed_inverse
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestUtils(unittest.TestCase):
    def test_embed_inverse(self):
        with nn.parameter_scope("embed_inverse"):
            x = nn.Variable((3, ))
            x.d[0] = 0
            x.d[1] = 1
            x.d[2] = 2
            with nn.auto_forward():
                embed = PF.embed(x, 3, 10)
                x = nn.Variable((3, 10))
                x2 = embed_inverse(embed, 3, 10)
            self.assertEqual(x2.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
