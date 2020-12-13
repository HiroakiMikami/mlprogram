import numpy as np

from mlprogram.builtins import Environment
from mlprogram.metrics import Accuracy


class TestAccuracy(object):
    def test_simple_case(self):
        acc = Accuracy()
        assert np.allclose(
            1.0,
            acc(Environment({"ground_truth": "str"}), "str"))
        assert np.allclose(
            0.0,
            acc(Environment({"ground_truth": "int"}), "str"))
