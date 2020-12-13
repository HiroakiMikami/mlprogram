import numpy as np
import torch

from mlprogram.languages.csg import Interpreter, show
from mlprogram.languages.csg.ast import Rectangle
from mlprogram.languages.csg.transforms import (
    AddTestCases,
    TransformInputs,
    TransformVariables,
)


class TestAddTestCases(object):
    def test_non_reference(self):
        f = AddTestCases(Interpreter(1, 1, 1, False))
        result = f(Rectangle(1, 1))
        assert len(result) == 1
        assert "#\n" == show(result[0][1])


class TestTransformInputs(object):
    def test_happy_path(self):
        f = TransformInputs()
        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 2, 2),
            f([(None, np.array([[True, False], [False, True]]))])
        )


class TestTransformVariables(object):
    def test_happy_path(self):
        f = TransformVariables()
        assert np.array_equal(
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]]).reshape(1, 1, 1, 2, 2),
            f(test_case_tensor=torch.rand(1, 1, 2, 2),
              variables=[[np.array([[True, False], [False, True]])]]
              )
        )
