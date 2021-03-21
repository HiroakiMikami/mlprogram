import numpy as np

from mlprogram.metrics import Accuracy


class TestAccuracy(object):
    def test_simple_case(self):
        acc = Accuracy()
        assert np.allclose(
            1.0,
            acc(expected="str", actual="str"))
        assert np.allclose(
            0.0,
            acc(expected="int", actual="str"))
