from mlprogram.builtins import Environment, Pick


class TestPick(object):
    def test_happy_path(self):
        pick = Pick("x")
        out = pick(Environment({"x": 10}))
        assert 10 == out

    def test_if_key_not_exist(self):
        pick = Pick("x")
        out = pick(Environment())
        assert out is None
