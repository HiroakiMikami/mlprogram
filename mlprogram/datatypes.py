from typing import Dict
from typing import Tuple
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TypeVar
from typing import Generic


K = TypeVar("K")
V = TypeVar("V")


class _Dict(Generic[K, V]):
    def __init__(self, values: Optional[Dict[K, V]] = None):
        values = values or {}
        self._values = {k: v for k, v in values.items()}
        self._mutable = True

    def mutable(self, f: bool = True) -> None:
        self._mutable = f

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: K) -> V:
        return self._values[key]

    def __setitem__(self, key: K, value: V):
        assert self._mutable
        self._values[key] = value

    def clear(self) -> None:
        assert self._mutable
        self._values.clear()

    def keys(self) -> Iterable[K]:
        return self._values.keys()

    def values(self) -> Iterable[V]:
        return self._values.values()

    def items(self) -> Iterable[Tuple[K, V]]:
        return self._values.items()

    def to_dict(self) -> Dict[K, V]:
        return {key: value for key, value in self._values.items()}

    def __str__(self) -> str:
        return f"_Dict({self._values}, mutable={self._mutable})"

    def __repr__(self) -> str:
        return f"_Dict({self._values}, mutable={self._mutable})"

    def __contains__(self, key: K) -> bool:
        return key in self._values


class Environment(object):
    input_prefix = "input"
    state_prefix = "state"
    output_prefix = "output"
    supervision_prefix = "supervision"
    prefixes = set([input_prefix, state_prefix, output_prefix,
                    supervision_prefix])

    def __init__(self,
                 inputs: Optional[Dict[str, Any]] = None,
                 states: Optional[Dict[str, Any]] = None,
                 outputs: Optional[Dict[str, Any]] = None,
                 supervisions: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._inputs = _Dict(inputs)
        self._states = _Dict(states)
        self._outputs = _Dict(outputs)
        self._supervisions = _Dict(supervisions)

    def _dict_from_prefix(self, prefix: str) -> _Dict[str, Any]:
        if prefix == Environment.input_prefix:
            return self.inputs
        if prefix == Environment.output_prefix:
            return self.outputs
        if prefix == Environment.state_prefix:
            return self.states
        if prefix == Environment.supervision_prefix:
            return self.supervisions
        raise RuntimeError(f"Invalid prefix {prefix}")

    @property
    def supervisions(self) -> _Dict[str, Any]:
        return self._supervisions

    @property
    def inputs(self) -> _Dict[str, Any]:
        return self._inputs

    @property
    def outputs(self) -> _Dict[str, Any]:
        return self._outputs

    @property
    def states(self) -> _Dict[str, Any]:
        return self._states

    def __setitem__(self, key: str, value: Any) -> None:
        prefix, key = Environment.parse_key(key)
        self._dict_from_prefix(prefix)[key] = value

    def __getitem__(self, key: str) -> Any:
        prefix, key = Environment.parse_key(key)
        return self._dict_from_prefix(prefix)[key]

    def clone(self):
        return Environment(
            _Dict(self.inputs.to_dict()),
            _Dict(self.states.to_dict()),
            _Dict(self.outputs.to_dict()),
            _Dict(self.supervisions.to_dict())
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for prefix in self.prefixes:
            for key, value in self._dict_from_prefix(prefix).items():
                out[f"{prefix}@{key}"] = value
        return out

    @staticmethod
    def parse_key(key: str) -> Tuple[str, str]:
        prefix, value = key.split("@")
        return prefix, value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Environment):
            return False
        return self.to_dict() == other.to_dict()

    def __str__(self) -> str:
        return f"Environment({self.to_dict()})"

    def __repr__(self) -> str:
        return f"Environment({self.to_dict()})"

    def mutable(self, inputs: bool = True, outputs: bool = True,
                states: bool = True, supervisions: bool = True) -> None:
        self.inputs.mutable(inputs)
        self.outputs.mutable(outputs)
        self.states.mutable(states)
        self.supervisions.mutable(supervisions)

    @staticmethod
    def create(values: Dict[str, Any]):
        env = Environment()
        for key, value in values.items():
            env[key] = value
        return env
