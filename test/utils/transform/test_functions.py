import unittest
from mlprogram.utils.transform import EvaluateGroundTruth
from mlprogram.utils.transform import NormalizeGroudTruth
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code):
        return int(code)

    def eval_references(self, code):
        return {stmt.reference: int(stmt.code) for stmt in code.statements}


class TestEvaluateGroundTruth(unittest.TestCase):
    def test_non_reference(self):
        f = EvaluateGroundTruth(MockInterpreter())
        self.assertEqual(1, f({"ground_truth": "1"})["input"])

    def test_reference(self):
        f = EvaluateGroundTruth(MockInterpreter(), reference=True)
        result = f({
            "ground_truth": SequentialProgram([
                Statement(Reference(0), "1"),
                Statement(Reference(1), "2")
            ])
        })
        self.assertEqual(2, result["input"])


class TestNormalizeGroundTruth(unittest.TestCase):
    def test_happy_path(self):
        f = NormalizeGroudTruth(lambda x: len(x))
        self.assertEqual(1, f({"ground_truth": [1]})["ground_truth"])

    def test_return_None(self):
        f = NormalizeGroudTruth(lambda x: None)
        self.assertEqual([1], f({"ground_truth": [1]})["ground_truth"])


if __name__ == "__main__":
    unittest.main()
