import torch
import numpy as np

from mlprogram import Environment

from mlprogram.languages.csg import Interpreter
from mlprogram.languages.csg import show
from mlprogram.languages.csg.ast import Rectangle
from mlprogram.languages.csg.transform import TransformCanvas
from mlprogram.languages.csg.transform import AddTestCases


class TestAddTestCases(object):
    def test_non_reference(self):
        f = AddTestCases(Interpreter(1, 1, 1, False))
        result = f(Environment(supervisions={"ground_truth": Rectangle(1, 1)}))
        assert "#\n" == show(result.inputs["test_case"][1])


class TestTransformCanvas(object):
    def test_happy_path(self):
        f = TransformCanvas()
        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 2, 2),
            f(Environment(inputs={
                "test_case": (None, np.array([[True, False], [False, True]]))
            })).states["test_case_tensor"]
        )

        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 2, 2),
            f(Environment(inputs={
                "test_case": np.array([[True, False], [False, True]]),
            },
                states={
                "variables": [np.array([[True, False], [False, True]])]
            })).states["variables_tensor"]
        )
