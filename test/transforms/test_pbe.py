import numpy as np
import torch

from mlprogram.builtins import Environment
from mlprogram.languages import Interpreter, Token
from mlprogram.transforms.pbe import ToEpisode


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
            {"test_cases": [(torch.tensor(1), torch.tensor(0))], "ground_truth": "01"},
            set(["ground_truth"])
        ))
        assert 2 == len(retval)
        assert len(retval[0]["test_cases"]) == 1
        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[0]["test_cases"][0]
        )
        assert [] == retval[0]["reference"]
        assert retval[0]["variables"] == []
        assert retval[0].is_supervision("ground_truth")
        assert '0' == retval[0]["ground_truth"]

        assert len(retval[1]["test_cases"]) == 1
        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[1]["test_cases"][0]
        )
        assert retval[1]["variables"] == [[torch.tensor(1)]]
        assert [Token(None, '0', '0')] == retval[1]["reference"]
        assert retval[1].is_supervision("ground_truth")
        assert '1' == retval[1]["ground_truth"]
