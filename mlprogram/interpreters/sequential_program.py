from typing import Generic, TypeVar, List, Any
from dataclasses import dataclass


Code = TypeVar("Code")


@dataclass
class Reference:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Reference):
            return self.name == other.name
        else:
            return False


@dataclass
class Statement(Generic[Code]):
    reference: Reference
    code: Code

    def __hash__(self) -> int:
        return hash((self.reference, self.code))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Statement):
            return self.reference == other.reference and \
                    self.code == other.code
        else:
            return False


class SequentialProgram(Generic[Code]):
    def __init__(self, stmts: List[Statement]):
        self._stmts = stmts

    @property
    def statements(self) -> List[Statement]:
        return self._stmts

    def __hash__(self) -> int:
        return hash(tuple(self.statements))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SequentialProgram):
            return self.statements == other.statements
        else:
            return False
