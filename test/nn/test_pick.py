import unittest

from mlprogram.nn import Pick


class TestPick(unittest.TestCase):
    def test_parameters(self):
        pick = Pick("")
        params = dict(pick.named_parameters())
        self.assertEqual(0, len(params))

    def test_happy_path(self):
        pick = Pick("x")
        out = pick({"x": 10})
        self.assertEqual(10, out)

    def test_if_key_not_exist(self):
        pick = Pick("x")
        out = pick({})
        self.assertEqual(None, out)


if __name__ == "__main__":
    unittest.main()
