import numpy as np

from mlprogram import Environment
from mlprogram.metrics import Accuracy


class TestAccuracy(object):
    def test_simple_case(self):
        acc = Accuracy()
        assert np.allclose(
            1.0,
            acc(Environment(supervisions={"ground_truth": "str"}), "str"))
        assert np.allclose(
            0.0,
            acc(Environment(supervisions={"ground_truth": "int"}), "str"))
