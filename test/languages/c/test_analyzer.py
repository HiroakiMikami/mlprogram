import unittest
from mlprogram.languages.c import Analyzer


class TestAnalyzer(unittest.TestCase):
    def test_without_errors(self):
        analyzer = Analyzer()
        self.assertEqual([], analyzer("int a = 0;"))

    def test_error(self):
        analyzer = Analyzer()
        self.assertEqual(1, len(analyzer("int a = 0")))

    def test_warning(self):
        analyzer = Analyzer()
        self.assertEqual(1, len(analyzer("int f(){return \"\";}")))

    def test_errors(self):
        analyzer = Analyzer()
        self.assertEqual(2, len(analyzer("b = 0;\nint a = 0")))


if __name__ == "__main__":
    unittest.main()
