import pytest
import torch

from mlprogram.builtins import Environment
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class TestEnv(object):
    def test_constructor(self) -> None:
        e = Environment()
        assert e.to_dict() == {}
        e["key"] = 0
        e["key2"] = 1
        e.mark_as_supervision("key2")
        assert e["key"] == 0
        with pytest.raises(AssertionError):
            e["key3"]
        assert e.to_dict() == {"key": 0, "key2": 1}

    def test_mark_as_supervision(self) -> None:
        with pytest.raises(AssertionError):
            e = Environment()
            e.mark_as_supervision("key2")

    def test_clone(self) -> None:
        e = Environment()
        e2 = e.clone()
        e["key"] = 0
        assert e2.to_dict() == {}

    def test_clone_without_supervision(self) -> None:
        e = Environment()
        e["key"] = 0
        e.mark_as_supervision("key")
        e2 = e.clone_without_supervision()
        assert e2.to_dict() == {}

    def test_get(self) -> None:
        e = Environment()
        e["key"] = 0
        assert e["key"] == 0

    def test_to(self) -> None:
        class X:
            def to(self, *args, **kwargs):
                self.args = (args, kwargs)
                return self

        e = Environment()
        e["key"] = X()
        e["x"] = torch.tensor(0)
        e["y"] = PaddedSequenceWithMask(torch.tensor(0.0), torch.tensor(True))
        e["z"] = 10
        e.to(device=torch.device("cpu"))
        assert e["key"].args == ((), {"device": torch.device("cpu")})
