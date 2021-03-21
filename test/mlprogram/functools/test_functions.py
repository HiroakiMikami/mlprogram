from collections import OrderedDict

from mlprogram.functools import Compose, Map, Sequence


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


def raise_exception(x):
    raise NotImplementedError()


class TestMap(object):
    def test_happy_path(self):
        f = Map(lambda x: x + 1)
        assert [3] == f([2])

    def test_multiprocessing(self):
        f = Map(add1, 1)
        assert [3] == f([2])

    def test_exception(self):
        f = Map(raise_exception)
        assert f([2]) == [None]


class TestSequence(object):
    def test_happy_path(self):
        f = Sequence(OrderedDict([("f0", lambda x: {"x": x["x"] + 1}),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        assert {"x": 6} == f({"x": 2})

    def test_f_return_none(self):
        f = Sequence(OrderedDict([("f0", lambda x: None),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        assert f({"x": 2}) is None
