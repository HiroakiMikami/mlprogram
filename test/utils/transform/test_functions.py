from mlprogram import Environment
from mlprogram.utils.transform import NormalizeGroundTruth


class TestNormalizeGroundTruth(object):
    def test_happy_path(self):
        f = NormalizeGroundTruth(lambda x: len(x))
        assert 1 == f(Environment(
            supervisions={"ground_truth": [1]}
        )).supervisions["ground_truth"]

    def test_return_None(self):
        f = NormalizeGroundTruth(lambda x: None)
        assert [1] == f(Environment(
            supervisions={"ground_truth": [1]}
        )).supervisions["ground_truth"]
