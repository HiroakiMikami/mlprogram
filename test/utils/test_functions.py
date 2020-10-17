import tempfile
import os
from collections import OrderedDict
from mlprogram import Environment
from mlprogram.utils \
    import Compose, Sequence, Map, Flatten, Pick, Threshold, save, load


class TestCompose(object):
    def test_happy_path(self):
        f = Compose(OrderedDict([("f", lambda x: x + 1),
                                 ("g", lambda y: y * 2)]))
        assert 6 == f(2)

    def test_value_is_none(self):
        f = Compose(OrderedDict([("f", lambda x: x + 1),
                                 ("g", lambda y: y * 2)]))
        assert f(None) is None

    def test_return_none(self):
        f = Compose(OrderedDict([("f", lambda x: None),
                                 ("g", lambda y: y * 2)]))
        assert f(2) is None


def add1(x):
    return x + 1


class TestMap(object):
    def test_happy_path(self):
        f = Map(lambda x: x + 1)
        assert [3] == f([2])

    def test_multiprocessing(self):
        f = Map(add1, 1)
        assert [3] == f([2])


class TestFlatten(object):
    def test_happy_path(self):
        f = Flatten()
        assert [2, 1, 2] == f([[2], [1, 2]])


class TestSequence(object):
    def test_happy_path(self):
        f = Sequence(OrderedDict([("f0", lambda x: {"x": x["x"] + 1}),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        assert {"x": 6} == f({"x": 2})

    def test_f_return_none(self):
        f = Sequence(OrderedDict([("f0", lambda x: None),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        assert f({"x": 2}) is None


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


class MockData:
    def __init__(self, x):
        self.x = x


class TestSave(object):
    def test_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file")
            obj = save(MockData(10), path)
            assert 10 == obj.x
            assert os.path.exists(path)


class TestLoad(object):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file")
            save(MockData(10), path)

            value = load(path)
            assert 10 == value.x
