import unittest
import tempfile
import torch
import torch.nn as nn
import os
import pytorch_pfn_extras as ppe

from mlprogram.pytorch_pfn_extras import SaveTopKModel


class TestSaveTopKModel(unittest.TestCase):
    def test_simple_case(self):
        model = nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ppe.training.ExtensionsManager(
                {}, {}, 1,
                out_dir=tmpdir,
                extensions=[],
                iters_per_epoch=1,
            )
            topk = SaveTopKModel(tmpdir, 2, "score", model)
            with manager.run_iteration():
                ppe.reporting.report({"score": 1.0})
                topk(manager)
            self.assertEqual(["model_0.pt"], os.listdir(tmpdir))

            with manager.run_iteration():
                ppe.reporting.report({"score": 2.0})
                topk(manager)
            self.assertEqual(["model_0.pt", "model_1.pt"],
                             sorted(os.listdir(tmpdir)))

            with manager.run_iteration():
                ppe.reporting.report({"score": 3.0})
                topk(manager)
            self.assertEqual(["model_1.pt", "model_2.pt"],
                             sorted(os.listdir(tmpdir)))

            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                topk(manager)
            self.assertEqual(["model_1.pt", "model_2.pt"],
                             sorted(os.listdir(tmpdir)))

            result = torch.load(os.path.join(tmpdir, "model_2.pt"))
            self.assertEqual(3.0, result["score"])

    def test_resume_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = nn.Linear(1, 1)
            manager = ppe.training.ExtensionsManager(
                {}, {}, 1,
                out_dir=tmpdir,
                extensions=[],
                iters_per_epoch=1,
            )
            topk = SaveTopKModel(tmpdir, 2, "score", model)
            with manager.run_iteration():
                ppe.reporting.report({"score": 1.0})
                topk(manager)
            with manager.run_iteration():
                ppe.reporting.report({"score": 2.0})
                topk(manager)

            topk = SaveTopKModel(tmpdir, 2, "score", model)
            with manager.run_iteration():
                ppe.reporting.report({"score": 3.0})
                topk(manager)
            self.assertEqual(["model_1.pt", "model_2.pt"],
                             sorted(os.listdir(tmpdir)))

            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                topk(manager)
            self.assertEqual(["model_1.pt", "model_2.pt"],
                             sorted(os.listdir(tmpdir)))


if __name__ == "__main__":
    unittest.main()
