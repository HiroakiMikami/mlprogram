import torch
import numpy as np
import unittest
from mlprogram.utils.transform.pbe_with_repl import ToEpisode, EvaluateCode
from mlprogram.languages import Token
from mlprogram.languages import Leaf
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code, input):
        return int(code) + input

    def eval_references(self, code, input):
        return {stmt.reference: int(stmt.code) + input
                for stmt in code.statements}


class TestToEpisode(unittest.TestCase):
    def test_happy_path(self):
        f = ToEpisode(remove_used_reference=False)
        retval = f({
            "input": (torch.tensor(1), torch.tensor(0)),
            "ground_truth": SequentialProgram([
                Statement(Reference(0), 0),
                Statement(Reference(1), 1)
            ])
        })
        self.assertEqual(2, len(retval))
        self.assertTrue(np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[0]["input"]
        ))
        self.assertEqual(
            SequentialProgram([Statement(Reference(0), 0)]),
            retval[0]["code"]
        )
        self.assertEqual(0, retval[0]["ground_truth"])
        self.assertEqual([], retval[0]["reference"])

        self.assertTrue(np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[1]["input"]
        ))
        self.assertEqual(
            SequentialProgram([Statement(Reference(0), 0),
                               Statement(Reference(1), 1)]),
            retval[1]["code"]
        )
        self.assertEqual(1, retval[1]["ground_truth"])
        self.assertEqual([Token(None, Reference(0), Reference(0))],
                         retval[1]["reference"])

    def test_remove_unused_reference(self):
        f = ToEpisode(to_ast=lambda x: Leaf("", Reference(0)),
                      remove_used_reference=True)
        retval = f({
            "input": (torch.tensor(1), torch.tensor(0)),
            "ground_truth": SequentialProgram([
                Statement(Reference(0), torch.tensor(0)),
                Statement(Reference(1), torch.tensor(1)),
                Statement(Reference(2), torch.tensor(2))
            ])
        })
        self.assertEqual(3, len(retval))
        self.assertTrue(np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[0]["input"]
        ))
        self.assertEqual(
            SequentialProgram([Statement(Reference(0), torch.tensor(0))]),
            retval[0]["code"]
        )
        self.assertEqual(0, retval[0]["ground_truth"])
        self.assertEqual([], retval[0]["reference"])

        self.assertTrue(np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[1]["input"]
        ))
        self.assertEqual(
            SequentialProgram([Statement(Reference(0), torch.tensor(0)),
                               Statement(Reference(1), torch.tensor(1))]),
            retval[1]["code"]
        )
        self.assertEqual(1, retval[1]["ground_truth"])
        self.assertEqual([], retval[1]["reference"])

        self.assertTrue(np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[2]["input"]
        ))
        self.assertEqual(
            SequentialProgram([Statement(Reference(0), torch.tensor(0)),
                               Statement(Reference(1), torch.tensor(1)),
                               Statement(Reference(2), torch.tensor(2))]),
            retval[2]["code"]
        )
        self.assertEqual(2, retval[2]["ground_truth"])
        self.assertEqual([Token(None, Reference(1), Reference(1))],
                         retval[2]["reference"])


class TestEvaluateCode(unittest.TestCase):
    def test_happy_path(self):
        f = EvaluateCode(MockInterpreter())
        output = f({
            "input": (1, None),
            "reference": [Token(None, Reference(1), Reference(1))],
            "code": SequentialProgram(
                [Statement(Reference(0), "0"), Statement(Reference(1), "1")]
            )
        })
        self.assertEqual([2], output["variables"])


if __name__ == "__main__":
    unittest.main()
