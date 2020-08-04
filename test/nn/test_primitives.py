import unittest

from mlprogram.nn import Add, Mul, Sub, Div, IntDiv, Neg


class TestAdd(unittest.TestCase):
    def test_parameters(self):
        op = Add()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = Add()
        out = op(10, 20)
        self.assertEqual(30, out)

    def test_kwargs(self):
        op = Add()
        out = op(x=10, y=20)
        self.assertEqual(30, out)


class TestMul(unittest.TestCase):
    def test_parameters(self):
        op = Mul()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = Mul()
        out = op(10, 20)
        self.assertEqual(200, out)

    def test_kwargs(self):
        op = Mul()
        out = op(x=10, y=20)
        self.assertEqual(200, out)


class TestSub(unittest.TestCase):
    def test_parameters(self):
        op = Sub()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = Sub()
        out = op(10, 20)
        self.assertEqual(-10, out)


class TestDiv(unittest.TestCase):
    def test_parameters(self):
        op = Div()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = Div()
        out = op(10, 20)
        self.assertEqual(0.5, out)


class TestIntDiv(unittest.TestCase):
    def test_parameters(self):
        op = IntDiv()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = IntDiv()
        out = op(10, 20)
        self.assertEqual(0, out)


class TestNeg(unittest.TestCase):
    def test_parameters(self):
        op = Neg()
        params = dict(op.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        op = Neg()
        out = op(10)
        self.assertEqual(-10, out)


if __name__ == "__main__":
    unittest.main()
