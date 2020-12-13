from mlprogram.builtins import Environment
from mlprogram.transforms import NormalizeGroundTruth


class TestNormalizeGroundTruth(object):
    def test_happy_path(self):
        f = NormalizeGroundTruth(lambda x: len(x))
        out = f(Environment({"ground_truth": [1]}, set(["ground_truth"])))
        assert out.is_supervision("ground_truth")
        assert 1 == out["ground_truth"]

    def test_return_None(self):
        f = NormalizeGroundTruth(lambda x: None)
        out = f(Environment({"ground_truth": [1]}, set(["ground_truth"])))
        assert out.is_supervision("ground_truth")
        assert [1] == out["ground_truth"]
