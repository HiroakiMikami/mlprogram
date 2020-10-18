import torch
import numpy as np
from mlprogram import Environment
from mlprogram.utils.transform.pbe_with_repl import ToEpisode
from mlprogram.languages import Token
from mlprogram.interpreters import Interpreter


class MockInterpreter(Interpreter):
    def eval(self, code, input):
        return int(code) + input


class MockExpander():
    def expand(self, code):
        return list(code)


class TestToEpisode(object):
    def test_happy_path(self):
        f = ToEpisode(MockInterpreter(), MockExpander())
        retval = f(Environment(
            inputs={"test_case": (torch.tensor(1), torch.tensor(0))},
            supervisions={"ground_truth": "01"}
        ))
        assert 2 == len(retval)
        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[0].inputs["test_case"]
        )
        assert [] == retval[0].states["reference"]
        assert retval[0].states["variables"] == []
        assert '0' == retval[0].supervisions["ground_truth"]

        assert np.array_equal(
            (torch.tensor(1), torch.tensor(0)),
            retval[1].inputs["test_case"]
        )
        assert retval[1].states["variables"] == [torch.tensor(1)]
        assert [Token(None, '0', '0')] == retval[1].states["reference"]
        assert '1' == retval[1].supervisions["ground_truth"]
