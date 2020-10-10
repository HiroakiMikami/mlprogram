from mlprogram.utils.transform import NormalizeGroudTruth


class TestNormalizeGroundTruth(object):
    def test_happy_path(self):
        f = NormalizeGroudTruth(lambda x: len(x))
        assert 1 == f({"ground_truth": [1]})["ground_truth"]

    def test_return_None(self):
        f = NormalizeGroudTruth(lambda x: None)
        assert [1] == f({"ground_truth": [1]})["ground_truth"]
