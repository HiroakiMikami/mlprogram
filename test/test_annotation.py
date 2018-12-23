import unittest
from src.annotation import Annotation, to_encoder_input
import os
import numpy as np


class TestAnnotation(unittest.TestCase):
    def test_annotation(self):
        a = Annotation(["word1", "word2", "unknown"], {})
        word_to_id = {"word1": 1, "word2": 2, "<unknown>": 3}
        result = to_encoder_input(a, word_to_id)
        self.assertEqual(result.query.tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
