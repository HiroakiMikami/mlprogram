import torch
import numpy as np
import unittest
from mlprogram.utils.transform.pbe_with_repl import ToEpisode
from mlprogram.utils import Reference, Token
from mlprogram.asts import Leaf


class TestPbeWithRepl(unittest.TestCase):
    def test_happy_path(self):
        f = ToEpisode(remove_used_reference=False)
        retval = f({
            "input": torch.tensor(0),
            "ground_truth": [
                (Reference(0), 0),
                (Reference(1), 1)
            ],
            "variables": {
                Reference(0): torch.tensor(0),
                Reference(1): torch.tensor(1)
            }
        })
        self.assertEqual(2, len(retval))
        self.assertTrue(np.array_equal(
            torch.tensor(0),
            retval[0]["input"]
        ))
        self.assertTrue(np.array_equal(
            torch.zeros(0,),
            retval[0]["variables"]
        ))
        self.assertEqual(0, retval[0]["ground_truth"])
        self.assertEqual([], retval[0]["reference"])

        self.assertTrue(np.array_equal(
            torch.tensor(0),
            retval[1]["input"]
        ))
        self.assertTrue(np.array_equal(
            torch.tensor([0]),
            retval[1]["variables"]
        ))
        self.assertEqual(1, retval[1]["ground_truth"])
        self.assertEqual([Token(None, Reference(0))], retval[1]["reference"])

    def test_remove_unused_reference(self):
        f = ToEpisode(to_ast=lambda x: Leaf("", Reference(0)),
                      remove_used_reference=True)
        retval = f({
            "input": torch.tensor(0),
            "ground_truth": [
                (Reference(0), torch.tensor(0)),
                (Reference(1), torch.tensor(1)),
                (Reference(2), torch.tensor(2))
            ],
            "variables": {
                Reference(0): torch.tensor(0),
                Reference(1): torch.tensor(1),
                Reference(2): torch.tensor(2),
            }
        })
        self.assertEqual(3, len(retval))
        self.assertTrue(np.array_equal(
            torch.tensor(0),
            retval[0]["input"]
        ))
        self.assertTrue(np.array_equal(
            torch.zeros(0,),
            retval[0]["variables"]
        ))
        self.assertEqual(0, retval[0]["ground_truth"])
        self.assertEqual([], retval[0]["reference"])

        self.assertTrue(np.array_equal(
            torch.tensor(0),
            retval[1]["input"]
        ))
        self.assertTrue(np.array_equal(
            torch.zeros(0,),
            retval[1]["variables"]
        ))
        self.assertEqual(1, retval[1]["ground_truth"])
        self.assertEqual([], retval[1]["reference"])

        self.assertTrue(np.array_equal(
            torch.tensor(0),
            retval[2]["input"]
        ))
        self.assertTrue(np.array_equal(
            torch.tensor([1]),
            retval[2]["variables"]
        ))
        self.assertEqual(2, retval[2]["ground_truth"])
        self.assertEqual([Token(None, Reference(1))], retval[2]["reference"])


if __name__ == "__main__":
    unittest.main()
