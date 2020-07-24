import unittest
from mlprogram.utils.transform.pbe_with_repl import ToEpisode
from mlprogram.utils import Reference, Token
from mlprogram.asts import Leaf


class TestPbeWithRepl(unittest.TestCase):
    def test_happy_path(self):
        f = ToEpisode(remove_used_reference=False)
        retval = f({
            "input": 0,
            "ground_truth": [
                (Reference(0), 0),
                (Reference(1), 1)
            ],
            "variables": {
                Reference(0): 0,
                Reference(1): 1
            }
        })
        self.assertEqual(2, len(retval))
        self.assertEqual(
            {
                "input": 0,
                "ground_truth": 0,
                "reference": [],
                "variables": []
            },
            retval[0]
        )
        self.assertEqual(
            {
                "input": 0,
                "ground_truth": 1,
                "reference": [Token(None, Reference(0))],
                "variables": [0]
            },
            retval[1]
        )

    def test_remove_unused_reference(self):
        f = ToEpisode(to_ast=lambda x: Leaf("", Reference(0)),
                      remove_used_reference=True)
        retval = f({
            "input": 0,
            "ground_truth": [
                (Reference(0), 0),
                (Reference(1), 1),
                (Reference(2), 2)
            ],
            "variables": {
                Reference(0): 0,
                Reference(1): 1,
                Reference(2): 2,
            }
        })
        self.assertEqual(3, len(retval))
        self.assertEqual(
            {
                "input": 0,
                "ground_truth": 0,
                "reference": [],
                "variables": []
            },
            retval[0]
        )
        self.assertEqual(
            {
                "input": 0,
                "ground_truth": 1,
                "reference": [],
                "variables": []
            },
            retval[1]
        )
        self.assertEqual(
            {
                "input": 0,
                "ground_truth": 2,
                "reference": [Token(None, Reference(1))],
                "variables": [1]
            },
            retval[2]
        )


if __name__ == "__main__":
    unittest.main()
