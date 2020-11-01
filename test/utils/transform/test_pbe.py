import numpy as np
import torch

from mlprogram import Environment
from mlprogram.languages import Interpreter, Token
from mlprogram.utils.transform.pbe import ToEpisode


class MockInterpreter(Interpreter):
    def eval(self, code, inputs):
        return [int(code) + input for input in inputs]


class MockExpander():
    def expand(self, code):
        return list(code)


class TestToEpisode(object):
    def test_happy_path(self):
        f = ToEpisode(MockInterpreter(), MockExpander())
        retval = f(Environment(
            inputs={"test_cases": [(torch.tensor(1), torch.tensor(0))]},
            supervisions={"ground_truth": "01"}
        ))
        assert 2 == len(retval)
        assert len(retval[0].inputs["test_cases"]) == 1
        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[0].inputs["test_cases"][0]
        )
        assert [] == retval[0].states["reference"]
        assert retval[0].states["variables"] == []
        assert '0' == retval[0].supervisions["ground_truth"]

        assert len(retval[1].inputs["test_cases"]) == 1
        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[1].inputs["test_cases"][0]
        )
        assert retval[1].states["variables"] == [[torch.tensor(1)]]
        assert [Token(None, '0', '0')] == retval[1].states["reference"]
        assert '1' == retval[1].supervisions["ground_truth"]
