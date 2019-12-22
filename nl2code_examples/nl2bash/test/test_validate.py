import torch
import unittest
from typing import List
from nl2prog.utils.nl2code import Progress, Candidate
from nl2code_examples.nl2bash._validate import bleu4
from nl2code_examples.nl2bash import validate, parse


class TestBleu4(unittest.TestCase):
    def test_bleu4(self):
        self.assertEqual(bleu4(["find ./ -name foo"], "find ./ -name foo"), 1)
        self.assertTrue(
            bleu4(["find ./ -name foo"], "find ./ -name bar") > 0.5)


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
            Candidate(0.0, parse("x=10")),
            Candidate(1.0, parse("echo foo `echo a`"))]
        synthesizer = MockSynthesizer([], candidates)
        result0 = validate([], ["x=10"], ["x=10"],
                           lambda x: torch.FloatTensor(len(x), 1), synthesizer)
        result1 = validate([], ["echo foo `echo a`"], ["echo foo $(echo a)"],
                           lambda x: torch.FloatTensor(len(x), 1), synthesizer)
        results = list([result0, result1])

        self.assertEqual(2, len(results))
        self.assertEqual([], results[0].query)
        self.assertEqual(["x=10"], results[0].ground_truth)
        self.assertAlmostEqual(results[0].top1_score.bleu, 0)
        self.assertAlmostEqual(results[0].top1_score.normalized_bleu, 0)
        self.assertFalse(results[0].top1_score.is_match)
        self.assertFalse(results[0].top1_score.is_exact_match)
        self.assertAlmostEqual(results[0].top3_score.bleu, 1)
        self.assertAlmostEqual(results[0].top3_score.normalized_bleu, 1)
        self.assertTrue(results[0].top3_score.is_match)
        self.assertTrue(results[0].top3_score.is_exact_match)
        self.assertEqual([], results[1].query)
        self.assertEqual(["echo foo `echo a`"], results[1].ground_truth)
        self.assertEqual(results[1].top1_score, results[1].top3_score)
        self.assertTrue(results[1].top3_score.bleu > 0.7)
        self.assertAlmostEqual(results[1].top3_score.normalized_bleu, 1)
        self.assertTrue(results[1].top3_score.is_match)
        self.assertFalse(results[1].top3_score.is_exact_match)


if __name__ == "__main__":
    unittest.main()
