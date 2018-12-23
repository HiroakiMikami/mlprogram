import unittest
from src.nn.layers.recurrent import lstm, bilstm

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np


class TestRecurrent(unittest.TestCase):
    def test_lstm(self):
        xs = nn.Variable((1, 2, 1))
        xs.d = [[[1], [0]]]
        mask = nn.Variable((1, 2))
        mask.d = [[1, 0]]
        with nn.parameter_scope("lstm"), nn.auto_forward():
            hs, cs = lstm(xs, mask, 1)

            self.assertEqual(hs.shape, (1, 2, 1))
            self.assertEqual(cs.shape, (1, 2, 1))
            self.assertTrue(np.all(hs.d[:, 0, :] == hs.d[:, 1, :]))
            self.assertTrue(np.all(cs.d[:, 0, :] == cs.d[:, 1, :]))

    def test_lstm_dropout(self):
        xs = nn.Variable((1, 2, 1))
        mask = nn.Variable((1, 2))
        mask.d = [[1, 0]]
        with nn.parameter_scope("lstm_dropout"), nn.auto_forward():
            hs_ref, cs_ref = lstm(xs * 0.9, mask, 1)
        with nn.parameter_scope("lstm_dropout"), nn.auto_forward():
            hs, cs = lstm(xs, mask, 1, dropout=0.1, train=False)

            self.assertTrue(np.allclose(hs.d, hs_ref.d))
            self.assertTrue(np.allclose(cs.d, cs_ref.d))

    def test_bilstm(self):
        xs = nn.Variable((1, 2, 1))
        xs.d = [[[1], [0]]]
        mask = nn.Variable((1, 2))
        mask.d = [[1, 0]]
        with nn.parameter_scope("bilstm"), nn.auto_forward():
            hs = bilstm(xs, mask, 1)
            self.assertEqual(hs.shape, (1, 2, 2))
        rxs = nn.Variable((1, 2, 1))
        rxs.d = [[[0], [1]]]
        rmask = nn.Variable((1, 2))
        rmask.d = [[0, 1]]
        with nn.parameter_scope("bilstm"), nn.auto_forward():
            with nn.parameter_scope("forward"):
                h1, _ = lstm(xs, mask, 1)
            with nn.parameter_scope("backward"):
                h2, _ = lstm(rxs, rmask, 1)
            h1 = h1.d
            h2 = h2.d[:, ::-1, :]
            hs_ref = np.concatenate([h1, h2], axis=2)

        self.assertTrue(np.allclose(hs.d, hs_ref))


if __name__ == "__main__":
    unittest.main()
