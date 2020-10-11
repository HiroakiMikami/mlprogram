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
        self._values = values
        self._mutable = True

    def mutable(self) -> None:
        self._mutable = True

    def immutable(self) -> None:
        self._mutable = False

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


class Environment(object):
    input_prefix = "input@"
    state_prefix = "state@"
    output_prefix = "output@"
    supervision_prefix = "supervision@"
    prefixes = set([input_prefix, state_prefix, output_prefix,
                    supervision_prefix])

    def __init__(self,
                 inputs: Optional[_Dict[str, Any]] = None,
                 states: Optional[_Dict[str, Any]] = None,
                 outputs: Optional[_Dict[str, Any]] = None,
                 supervisions: Optional[_Dict[str, Any]] = None,
                 batch_size: Optional[int] = None):
        super().__init__()
        inputs = _Dict(inputs)
        states = _Dict(states)
        outputs = _Dict(outputs)
        supervisions = _Dict(supervisions)
        self._inputs = inputs
        self._states = states
        self._outputs = outputs
        self._supervisions = supervisions
        self._batch_size = batch_size

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
    def batch_size(self):
        return self._batch_size

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
        prefix, key = key.split("@")
        prefix = prefix + "@"
        self._dict_from_prefix(prefix)[key] = value

    def __getitem__(self, key: str) -> Any:
        prefix, key = key.split("@")
        prefix = prefix + "@"
        return self._dict_from_prefix(prefix)[key]

    def clone(self):
        return Environment(
            _Dict(self.inputs.to_dict()),
            _Dict(self.states.to_dict()),
            _Dict(self.outputs.to_dict()),
            _Dict(self.supervisions.to_dict()),
            self._batch_size
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for prefix in self.prefixes:
            for key, value in self._dict_from_prefix(prefix).items():
                out[f"{prefix}{key}"] = value
        return out
