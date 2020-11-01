import numpy as np
import torch

from mlprogram import Environment
from mlprogram.languages.csg import Interpreter, show
from mlprogram.languages.csg.ast import Rectangle
from mlprogram.languages.csg.transform import AddTestCases, TransformCanvas


class TestAddTestCases(object):
    def test_non_reference(self):
        f = AddTestCases(Interpreter(1, 1, 1, False))
        result = f(Environment(supervisions={"ground_truth": Rectangle(1, 1)}))
        assert len(result.inputs["test_cases"]) == 1
        assert "#\n" == show(result.inputs["test_cases"][0][1])


class TestTransformCanvas(object):
    def test_happy_path(self):
        f = TransformCanvas()
        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 2, 2),
            f(Environment(inputs={
                "test_cases":
                    [(None, np.array([[True, False], [False, True]]))]
            })).states["test_case_tensor"]
        )

        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 1, 2, 2),
            f(Environment(inputs={
                "test_cases": [np.array([[True, False], [False, True]])],
            },
                states={
                "variables": [[np.array([[True, False], [False, True]])]]
            })).states["variables_tensor"]
        )
