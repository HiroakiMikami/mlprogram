import unittest
import torch
from torch.nn import LSTMCell as pytorch_LSTMCell
import numpy as np

from nl2code.nn import LSTMCell


class TestLSTMCell(unittest.TestCase):
    def test_parameters(self):
        cell = LSTMCell(2, 3)
        ref = pytorch_LSTMCell(2, 3)
        params = dict(cell.named_parameters())
        ref_params = dict(ref.named_parameters())
        self.assertEqual(set(ref_params.keys()), set(params.keys()))
        for key in params.keys():
            self.assertEqual(ref_params[key].shape, params[key].shape)

    def test_shape(self):
        cell = LSTMCell(2, 3)
        h_1, c_1 = cell(torch.FloatTensor(5, 2))
        self.assertEqual((5, 3), h_1.shape)
        self.assertEqual((5, 3), c_1.shape)

    def test_init(self):
        cell = LSTMCell(2, 3)
        self.assertTrue(np.all(cell.bias_ih[3:6].detach().numpy() == 1))

    def test_forward_with_bias(self):
        cell = LSTMCell(2, 3)
        ref = pytorch_LSTMCell(2, 3)
        ref_params = dict(ref.named_parameters())
        for key, param in cell.named_parameters():
            ref_params[key].data = param.data
        x = torch.FloatTensor(5, 2)

        h, c = cell(x)
        h_ref, c_ref = ref(x)
        self.assertTrue(np.allclose(h_ref.detach().numpy(),
                                    h.detach().numpy()))
        self.assertTrue(np.allclose(c_ref.detach().numpy(),
                                    c.detach().numpy()))

    def test_forward_without_bias(self):
        cell = LSTMCell(2, 3, bias=False)
        ref = pytorch_LSTMCell(2, 3, bias=False)
        ref_params = dict(ref.named_parameters())
        for key, param in cell.named_parameters():
            ref_params[key].data = param.data
        x = torch.FloatTensor(5, 2)

        h, c = cell(x)
        h_ref, c_ref = ref(x)
        self.assertTrue(np.allclose(h_ref.detach().numpy(),
                                    h.detach().numpy()))
        self.assertTrue(np.allclose(c_ref.detach().numpy(),
                                    c.detach().numpy()))

    def test_dropout_in_eval_mode(self):
        cell = LSTMCell(2, 3, bias=False, dropout=0.5)
        cell.eval()
        ref = pytorch_LSTMCell(2, 3, bias=False)
        ref_params = dict(ref.named_parameters())
        for key, param in cell.named_parameters():
            ref_params[key].data = param.data
        x = torch.FloatTensor(5, 2)

        h, c = cell(x)
        h_ref, c_ref = ref(x)
        self.assertTrue(np.allclose(h_ref.detach().numpy(),
                                    h.detach().numpy()))
        self.assertTrue(np.allclose(c_ref.detach().numpy(),
                                    c.detach().numpy()))


if __name__ == "__main__":
    unittest.main()
