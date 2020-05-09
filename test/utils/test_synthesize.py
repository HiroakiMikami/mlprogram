import unittest
import ast
from typing import List
from nl2prog.language.python import to_ast
from nl2prog.synthesizer import Progress, Candidate, synthesize as _synthesize


class TestSynthesize(unittest.TestCase):
    def test_simple_case(self):
        class MockSynthesizer:
            def __init__(self, progress: List[Progress],
                         candidates: List[Candidate]):
                self._progress = progress
                self._candidates = candidates

            def synthesize(self, query: str):
                yield self._candidates, self._progress

        candidates = [
            Candidate(0.0, to_ast(ast.parse("x = 10"))),
            Candidate(1.0, to_ast(ast.parse("x = 20")))]
        synthesizer = MockSynthesizer([], candidates)
        progress, results = _synthesize("", synthesizer)
        self.assertEqual([[]], progress)
        self.assertEqual(
            [candidates[1].ast, candidates[0].ast],
            results
        )


if __name__ == "__main__":
    unittest.main()
