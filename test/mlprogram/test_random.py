import numpy as np

from mlprogram.random import split


class TestSplit(object):
    def test_happy_path(self):
        ns = split(np.random.RandomState(0), 10, 3, 1e-5)
        assert 10 == sum(ns)
