from typing import Any, Dict, Iterable, Optional, Set, Tuple


class Environment(object):
    def __init__(self, values: Optional[Dict[str, Any]] = None,
                 supervisions: Optional[Set[str]] = None):
        values = values or {}
        supervisions = supervisions or set()
        assert all([key in values for key in supervisions])
        self._values: Dict[str, Any] = values
        self._supervisions: Set[str] = supervisions

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: str) -> Any:
        assert key in self._values
        return self._values[key]

    def __setitem__(self, key: str, value: Any):
        self._values[key] = value

    def mark_as_supervision(self, key: str) -> None:
        assert key in self._values
        self._supervisions.add(key)

    def is_supervision(self, key: str) -> bool:
        return key in self._supervisions

    def clear(self) -> None:
        self._values.clear()
        self._supervisions.clear()

    def keys(self) -> Iterable[str]:
        return self._values.keys()

    def values(self) -> Iterable[Any]:
        return self._values.values()

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self._values.items()

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def clone_without_supervision(self):
        return Environment({
            key: value for key, value in self._values.items()
            if key not in self._supervisions
        })

    def clone(self):
        return Environment(dict(self._values), set(self._supervisions))

    def to_dict(self) -> Dict[str, Any]:
        return {
            key if key not in self._supervisions else f"{key}": value
            for key, value in self._values.items()
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Environment):
            return False
        return self.to_dict() == other.to_dict()

    def to(self, *args, **kwargs) -> None:
        def _to(x: Any):
            if hasattr(x, "to"):
                return x.to(*args, **kwargs)
            else:
                return x
        self._values = {key: _to(x) for key, x in self._values.items()}

    def __str__(self) -> str:
        return f"Environment(${str(self.to_dict())})"

    def __repr__(self) -> str:
        return f"Environment(values={str(self._values)}, "
        f"supervisions={str(self._supervisions)})"
