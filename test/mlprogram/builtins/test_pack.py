from mlprogram.builtins import Environment, Pack


class TestPack(object):
    def test_happy_path(self):
        pack = Pack("x")
        out = pack(10)
        assert out == Environment({"x": 10})

    def test_mark_as_supervision(self):
        pack = Pack("x", is_supervision=True)
        out = pack(10)
        assert out == Environment({"x": 10}, set(["x"]))
