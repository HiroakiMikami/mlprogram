from mlprogram.builtins.operators import Add, Mul, Sub, Div, IntDiv, Neg


class TestAdd(object):
    def test_happy_path(self):
        op = Add()
        out = op(10, 20)
        assert 30 == out

    def test_kwargs(self):
        op = Add()
        out = op(x=10, y=20)
        assert 30 == out


class TestMul(object):
    def test_happy_path(self):
        op = Mul()
        out = op(10, 20)
        assert 200 == out

    def test_kwargs(self):
        op = Mul()
        out = op(x=10, y=20)
        assert 200 == out


class TestSub(object):
    def test_happy_path(self):
        op = Sub()
        out = op(10, 20)
        assert -10 == out


class TestDiv(object):
    def test_happy_path(self):
        op = Div()
        out = op(10, 20)
        assert 0.5 == out


class TestIntDiv(object):
    def test_happy_path(self):
        op = IntDiv()
        out = op(10, 20)
        assert 0 == out


class TestNeg(object):
    def test_happy_path(self):
        op = Neg()
        out = op(10)
        assert -10 == out
