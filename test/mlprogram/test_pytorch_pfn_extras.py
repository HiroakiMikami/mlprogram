import os
import tempfile

import pytest
import pytorch_pfn_extras as ppe
import torch
import torch.nn as nn

from mlprogram.pytorch_pfn_extras import SaveTopKModel, StopByThreshold


class TestSaveTopKModel(object):
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
            assert ["model_0.pt"] == os.listdir(tmpdir)

            with manager.run_iteration():
                ppe.reporting.report({"score": 2.0})
                topk(manager)
            assert ["model_0.pt", "model_1.pt"] == sorted(os.listdir(tmpdir))

            with manager.run_iteration():
                ppe.reporting.report({"score": 3.0})
                topk(manager)
            assert ["model_1.pt", "model_2.pt"] == sorted(os.listdir(tmpdir))

            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                topk(manager)
            assert ["model_1.pt", "model_2.pt"] == sorted(os.listdir(tmpdir))

            result = torch.load(os.path.join(tmpdir, "model_2.pt"))
            assert 3.0 == result["score"]

    def test_minimize(self):
        model = nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ppe.training.ExtensionsManager(
                {}, {}, 1,
                out_dir=tmpdir,
                extensions=[],
                iters_per_epoch=1,
            )
            topk = SaveTopKModel(tmpdir, 1, "score", model, maximize=False)
            with manager.run_iteration():
                ppe.reporting.report({"score": 1.0})
                topk(manager)
            assert ["model_0.pt"] == os.listdir(tmpdir)

            with manager.run_iteration():
                ppe.reporting.report({"score": 2.0})
                topk(manager)
            assert ["model_0.pt"] == sorted(os.listdir(tmpdir))

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
            assert ["model_1.pt", "model_2.pt"] == sorted(os.listdir(tmpdir))

            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                topk(manager)
            assert ["model_1.pt", "model_2.pt"] == sorted(os.listdir(tmpdir))


class TestStopByThreshold(object):
    def test_simple_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ppe.training.ExtensionsManager(
                {}, {}, 1,
                out_dir=tmpdir,
                extensions=[],
                iters_per_epoch=1,
            )
            stop = StopByThreshold("score", 0.5)
            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                stop(manager)

            with manager.run_iteration():
                ppe.reporting.report({"score": 1.0})
                with pytest.raises(RuntimeError):
                    stop(manager)

    def testminimizesimple_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ppe.training.ExtensionsManager(
                {}, {}, 1,
                out_dir=tmpdir,
                extensions=[],
                iters_per_epoch=1,
            )
            stop = StopByThreshold("score", 0.5, maximize=False)
            with manager.run_iteration():
                ppe.reporting.report({"score": 1.0})
                stop(manager)

            with manager.run_iteration():
                ppe.reporting.report({"score": 0.0})
                with pytest.raises(RuntimeError):
                    stop(manager)
