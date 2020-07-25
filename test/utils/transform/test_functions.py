import unittest
from mlprogram.utils import Reference
from mlprogram.utils.transform \
    import RandomChoice, EvaluateGroundTruth, NormalizeGroudTruth
from mlprogram.interpreters import Interpreter


class TestRandomChoice(unittest.TestCase):
    def test_choice(self):
        transform = RandomChoice()
        x = transform({"x": [0, 1], "y": [0, 1]})
        self.assertTrue(isinstance(x["x"], int))
        self.assertTrue(isinstance(x["y"], int))


class MockInterpreter(Interpreter):
    def eval(self, code):
        return int(code)

    def eval_references(self, code):
        return {ref: int(code) for ref, code in code}


class TestEvaluateGroundTruth(unittest.TestCase):
    def test_non_reference(self):
        f = EvaluateGroundTruth(MockInterpreter())
        self.assertEqual(1, f({"ground_truth": "1"})["test_case"])

    def test_reference(self):
        f = EvaluateGroundTruth(MockInterpreter())
        result = f({
            "ground_truth": [
                (Reference(0), "1"),
                (Reference(1), "2")
            ]
        })
        self.assertEqual(2, result["test_case"])
        self.assertEqual({Reference(0): 1, Reference(1): 2},
                         result["variables"])


class TestNormalizeGroundTruth(unittest.TestCase):
    def test_happy_path(self):
        f = NormalizeGroudTruth(lambda x: len(x))
        self.assertEqual([1], f({"ground_truth": [[1]]})["ground_truth"])

    def test_return_None(self):
        f = NormalizeGroudTruth(lambda x: None)
        self.assertEqual([[1]], f({"ground_truth": [[1]]})["ground_truth"])


if __name__ == "__main__":
    unittest.main()
