import pytest
from mlprogram.datatypes import _Dict
from mlprogram import Environment


class Test_Dict(object):
    def test_use_as_dict(self):
        d = _Dict()
        d["0"] = 0
        d["1"] = 1
        assert len(d) == 2
        assert d["0"] == 0
        assert d["1"] == 1
        assert set(d.keys()) == set(["0", "1"])
        assert set(d.values()) == set([0, 1])
        assert set(d.items()) == set([("0", 0), ("1", 1)])
        d.clear()
        assert len(d) == 0

    def test_to_dict(self):
        d = _Dict()
        d["0"] = 0
        d["1"] = 1
        assert d.to_dict() == {"0": 0, "1": 1}
        d.to_dict()["2"] = 2
        assert d.to_dict() == {"0": 0, "1": 1}

    def test_immutable(self):
        d = _Dict()
        d["0"] = 0
        d["1"] = 1
        d.mutable(False)
        with pytest.raises(AssertionError):
            d["2"] = 2
        with pytest.raises(AssertionError):
            d.clear()
        d.mutable()
        d["2"] = 2


class TestEnvironment(object):
    def test_constructor(self):
        e = Environment()
        e.inputs["0"] = 0
        assert e.inputs.to_dict() == {"0": 0}
        assert e.outputs.to_dict() == {}

    def test_to_dict(self):
        e = Environment()
        e.inputs["0"] = 0
        e.states["1"] = 1
        e.outputs["2"] = 2
        e.supervisions["3"] = 3
        assert e.to_dict() == {"input@0": 0, "state@1": 1, "output@2": 2,
                               "supervision@3": 3}

    def test_clone(self):
        e0 = Environment()
        e1 = e0.clone()
        e1.inputs["0"] = 0
        e1.states["1"] = 1
        e1.outputs["2"] = 2
        e1.supervisions["3"] = 3
        assert e0.to_dict() == {}
        assert e1.to_dict() == {"input@0": 0, "state@1": 1, "output@2": 2,
                                "supervision@3": 3}

    def test_setitem(self):
        e = Environment()
        e["input@0"] = 0
        assert e["input@0"] == 0
        assert e.inputs.to_dict() == {"0": 0}
        assert e.outputs.to_dict() == {}
