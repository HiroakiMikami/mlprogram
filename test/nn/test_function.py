
from mlprogram.nn import Function


class TestFunction(object):
    def test_parameters(self):
        f = Function(lambda x: x)
        params = dict(f.named_parameters())
        assert 0 == len(params)

    def test_happy_path(self):
        f = Function(lambda x: x)
        out = f(10)
        assert 10 == out
