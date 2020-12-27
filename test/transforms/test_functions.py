from mlprogram.transforms import NormalizeGroundTruth


class TestNormalizeGroundTruth(object):
    def test_happy_path(self):
        f = NormalizeGroundTruth(lambda x: len(x))
        out = f(ground_truth=[1])
        assert 1 == out

    def test_return_None(self):
        f = NormalizeGroundTruth(lambda x: None)
        out = f(ground_truth=[1])
        assert [1] == out
