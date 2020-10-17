from mlprogram import Environment
from mlprogram.utils import Flatten, Pick, Threshold


class TestFlatten(object):
    def test_happy_path(self):
        f = Flatten()
        assert [2, 1, 2] == f([[2], [1, 2]])


class TestThreshold(object):
    def test_happy_path(self):
        f = Threshold(0.6)
        assert f(0.7)
        assert not f(0.5)


class TestPick(object):
    def test_happy_path(self):
        pick = Pick("input@x")
        out = pick(Environment(inputs={"x": 10}))
        assert 10 == out

    def test_if_key_not_exist(self):
        pick = Pick("input@x")
        out = pick(Environment())
        assert out is None
