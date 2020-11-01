from typing import Any


class Kinds:
    class LineNumber:
        """
        The kind represents line-number
        """
        _instance = None

        def __hash__(self) -> int:
            return hash(str(self))

        def __eq__(self, rhs: Any) -> bool:
            return isinstance(rhs, Kinds.LineNumber)

        def __str__(self) -> str:
            return "<LineNumber>"

        def __repr__(self) -> str:
            return "<LineNumber>"

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
