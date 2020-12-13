from mlprogram.builtins import Flatten, Threshold


class TestFlatten(object):
    def test_happy_path(self):
        f = Flatten()
        assert [2, 1, 2] == f([[2], [1, 2]])


class TestThreshold(object):
    def test_happy_path(self):
        f = Threshold(0.6)
        assert f(0.7)
        assert not f(0.5)
