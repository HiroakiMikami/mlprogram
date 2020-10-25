from mlprogram.languages.linediff import IsSubtype


class TestIsSubType(object):
    def test_happy_path(self):
        assert IsSubtype()("Delta", "Delta")
        assert IsSubtype()("Insert", "Delta")
