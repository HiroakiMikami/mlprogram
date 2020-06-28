import unittest
import tempfile
import os
from collections import OrderedDict
from mlprogram.utils import Compose, Sequence, save, load


class TestCompose(unittest.TestCase):
    def test_happy_path(self):
        f = Compose(OrderedDict([("f", lambda x: x + 1),
                                 ("g", lambda y: y * 2)]))
        self.assertEqual(6, f(2))

    def test_value_is_none(self):
        f = Compose(OrderedDict([("f", lambda x: x + 1),
                                 ("g", lambda y: y * 2)]))
        self.assertEqual(None, f(None))

    def test_return_none(self):
        f = Compose(OrderedDict([("f", lambda x: None),
                                 ("g", lambda y: y * 2)]))
        self.assertEqual(None, f(2))


class TestSequence(unittest.TestCase):
    def test_happy_path(self):
        f = Sequence(OrderedDict([("f0", lambda x: {"x": x["x"] + 1}),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        self.assertEqual({"x": 6}, f({"x": 2}))

    def test_f_return_none(self):
        f = Sequence(OrderedDict([("f0", lambda x: None),
                                  ("f1", lambda x: {"x": x["x"] * 2})]))
        self.assertEqual(None, f({"x": 2}))


class MockData:
    def __init__(self, x):
        self.x = x


class TestSave(unittest.TestCase):
    def test_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file")
            obj = save(MockData(10), path)
            self.assertEqual(10, obj.x)
            self.assertTrue(os.path.exists(path))


class TestLoad(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "file")
            save(MockData(10), path)

            value = load(path)
            self.assertEqual(10, value.x)


if __name__ == "__main__":
    unittest.main()
