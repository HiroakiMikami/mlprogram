import torch
import unittest
from typing import List
from nl2code.language.python import to_ast
from nl2code import Progress, Candidate
from examples.django._validate import bleu4
from examples.django import validate, parse


class TestBleu4(unittest.TestCase):
    def test_bleu4(self):
        self.assertEqual(bleu4("def f():\n  pass\n", "def f():\n  pass\n"), 1)
        self.assertTrue(
            bleu4("def f():\n  pass\n", "def f(arg):\n  pass\n") > 0.4)
        self.assertTrue(
            bleu4("def f():\n  pass\n", "def f(arg):\n  pass\n") < 0.5)


class TestValidate(unittest.TestCase):
    def test_simple_case(self):
        class MockSynthesizer:
            def __init__(self, progress: List[Progress],
                         candidates: List[Candidate]):
                self._progress = progress
                self._candidates = candidates

            def synthesize(self, query: List[str],
                           embeddings: torch.FloatTensor):
                yield self._candidates, self._progress

        candidates = [
            Candidate(0.0, to_ast(parse("x = 10"))),
            Candidate(1.0, to_ast(parse("x = 20")))]
        synthesizer = MockSynthesizer([], candidates)
        result0 = validate([], [], "\nx = 20\n",
                           lambda x: torch.FloatTensor(len(x), 1), synthesizer)
        result1 = validate([], [], "\nx = 10\n",
                           lambda x: torch.FloatTensor(len(x), 1), synthesizer)
        results = list([result0, result1])

        self.assertEqual(2, len(results))
        self.assertEqual([], results[0].query)
        self.assertEqual("\nx = 20\n", results[0].ground_truth)
        self.assertAlmostEqual(1.0, results[0].bleu4)
        self.assertTrue(results[0].is_match)
        self.assertEqual(2, len(results))
        self.assertEqual([], results[1].query)
        self.assertEqual("\nx = 10\n", results[1].ground_truth)
        self.assertTrue(results[1].bleu4 < 0.7)
        self.assertFalse(results[1].is_match)


if __name__ == "__main__":
    unittest.main()
