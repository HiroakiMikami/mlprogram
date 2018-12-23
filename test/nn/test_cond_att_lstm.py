import unittest
from src.nn.cond_att_lstm import cond_att_lstm

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestCondAttLSTM(unittest.TestCase):
    def test_cond_att_lstm(self):
        xs = nn.Variable((1, 2, 1))
        xs.d = [[[1], [1]]]
        parent_index = nn.Variable((1, 2))
        parent_index.d = [[-1, 0]]
        mask = nn.Variable((1, 2))
        mask.d = [[1, 1]]
        context = nn.Variable((1, 3, 1))
        context.d = [[[1], [2], [0]]]
        context_mask = nn.Variable((1, 3))
        context_mask.d = [[1, 1, 0]]
        with nn.parameter_scope("cond_att_lstm"), nn.auto_forward():
            hs, cs, ctx, h = cond_att_lstm(xs, parent_index, mask, context,
                                           context_mask, 1, 1)

        self.assertEqual(hs.shape, (1, 2, 1))
        self.assertEqual(cs.shape, (1, 2, 1))
        self.assertEqual(ctx.shape, (1, 2, 1))
        self.assertEqual(h.shape, (1, 3, 1))

        self.assertTrue(np.all(h.d[:, 0, :] == 0))
        self.assertTrue(np.allclose(h.d[:, 1:, :], hs.d))


if __name__ == "__main__":
    unittest.main()
