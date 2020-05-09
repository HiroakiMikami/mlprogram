import unittest
import tempfile
import torch
import torch.nn as nn
import os

from mlprogram.utils import TopKModel


class TestTopKModel(unittest.TestCase):
    def test_simple_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            topk = TopKModel(2, tmpdir)
            model = nn.Linear(1, 1)
            topk.save(1.0, "1", model)
            self.assertEqual(["model_1.pt"], os.listdir(tmpdir))
            topk.save(2.0, "2", model)
            self.assertEqual(["model_1.pt", "model_2.pt"],
                             os.listdir(tmpdir))
            topk.save(3.0, "3", model)
            self.assertEqual(["model_2.pt", "model_3.pt"],
                             os.listdir(tmpdir))
            topk.save(0.0, "0", model)
            self.assertEqual(["model_2.pt", "model_3.pt"],
                             os.listdir(tmpdir))

            result = torch.load(os.path.join(tmpdir, "model_3.pt"))
            self.assertEqual(3.0, result["score"])

    def test_resume_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            topk = TopKModel(2, tmpdir)
            model = nn.Linear(1, 1)
            topk.save(1.0, "1", model)
            topk.save(2.0, "2", model)

            topk = TopKModel(2, tmpdir)
            topk.save(3.0, "3", model)
            self.assertEqual(["model_2.pt", "model_3.pt"],
                             os.listdir(tmpdir))
            topk.save(0.0, "0", model)
            self.assertEqual(["model_2.pt", "model_3.pt"],
                             os.listdir(tmpdir))


if __name__ == "__main__":
    unittest.main()
