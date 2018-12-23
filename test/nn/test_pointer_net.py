import unittest
from src.nn.pointer_net import pointer_net
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestPointerNet(unittest.TestCase):
    def test_pointer_net(self):
        query_embed = nn.Variable((2, 2, 1))
        query_embed.d = [[[0], [1]], [[1], [1]]]
        query_embed_mask = nn.Variable((2, 2))
        query_embed_mask.d = [[0, 1], [1, 1]]
        decoder_states = nn.Variable((2, 3, 1))
        decoder_states.d[:] = [[[0], [1], [2]]]
        with nn.parameter_scope("pointer_net"), nn.auto_forward():
            output = pointer_net(query_embed, query_embed_mask, decoder_states,
                                 2)
            self.assertTrue(np.allclose(output.d[0, :, 0], 0))
            self.assertTrue(np.allclose(output.d[0, :, 1], 1))
            self.assertTrue(np.allclose(output.d[1, :, :], 0.5))


if __name__ == "__main__":
    unittest.main()
