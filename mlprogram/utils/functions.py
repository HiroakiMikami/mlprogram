from typing import TypeVar, Generic, Optional, Callable

V0 = TypeVar("V0")
V1 = TypeVar("V1")
V2 = TypeVar("V2")


class Compose(Generic[V0, V1, V2]):
    def __init__(self, f: Callable[[V0], Optional[V1]],
                 g: Callable[[V1], Optional[V2]]):
        self.f = f
        self.g = g

    def __call__(self, value: Optional[V0]) -> Optional[V2]:
        if value is None:
            return None
        v1 = self.f(value)
        if v1 is None:
            return None
        return self.g(v1)
