import unittest
from src.nn.layers.embeddings import embedding
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestEmbeddings(unittest.TestCase):
    def test_embeddings(self):
        x = nn.Variable((2, ))
        x.d = [0, 1]
        with nn.parameter_scope("embeddings"):
            output = embedding(x, 2, 3)
            output.forward()

        with nn.parameter_scope("embeddings"), nn.auto_forward():
            output_ref = PF.embed(x, 2, 3)
            self.assertTrue(np.allclose(output.d, output_ref.d))

    def test_embeddings_mask_zero(self):
        x = nn.Variable((2, ))
        x.d = [0, 1]
        with nn.parameter_scope("embeddings"):
            output, mask = embedding(x, 2, 3, mask_zero=True)
            output.forward()
            mask.forward()
            self.assertTrue(np.all(mask.d == [0, 1]))

    def test_embeddings_parameter_shared(self):
        x = nn.Variable((2, ))
        x.d = [0, 1]
        with nn.parameter_scope("embeddings"):
            output = embedding(x, 2, 3)
            output.forward()

        with nn.parameter_scope("embeddings"):
            for v in nn.get_parameters().values():
                v.d = 1
            output = embedding(x, 2, 3)
            self.assertTrue(np.all(nn.get_parameters()["embed/W"].d == 1))


if __name__ == "__main__":
    unittest.main()
