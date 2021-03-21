from mlprogram.languages.c import Analyzer


class TestAnalyzer(object):
    def test_without_errors(self):
        analyzer = Analyzer()
        assert [] == analyzer("int a = 0;")

    def test_error(self):
        analyzer = Analyzer()
        assert 1 == len(analyzer("int a = 0"))

    def test_warning(self):
        analyzer = Analyzer()
        assert 1 == len(analyzer("int f(){return \"\";}"))

    def test_errors(self):
        analyzer = Analyzer()
        assert 2 == len(analyzer("b = 0;\nint a = 0"))
