import unittest
from src.nn.layers.core import dense
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestCore(unittest.TestCase):
    def test_dense(self):
        x = nn.Variable((1, 2))
        x.d = [[0, 1]]
        with nn.parameter_scope("dense"), nn.auto_forward():
            output = dense(x, 3)
            self.assertTrue(np.all(nn.get_parameters()["affine/b"].d == 0))
        with nn.parameter_scope("dense"), nn.auto_forward():
            output_ref = F.tanh(PF.affine(x, 3))
            self.assertTrue(np.allclose(output.d, output_ref.d))


if __name__ == "__main__":
    unittest.main()
