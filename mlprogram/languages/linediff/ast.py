from dataclasses import dataclass
from typing import List
from typing import Union


class Delta:
    def get_line_number(self) -> int:
        raise NotImplementedError

    def get_type_name(self) -> str:
        raise NotImplementedError


@dataclass
class Insert(Delta):
    line_number: int
    value: str

    def get_line_number(self) -> int:
        return self.line_number

    def get_type_name(self) -> str:
        return "Insert"

    def __hash__(self) -> int:
        return hash((self.line_number, self.value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Insert):
            return False
        return self.line_number == other.line_number and \
            self.value == other.value

    def __str__(self) -> str:
        return f"Insert({self.line_number}, {self.value})"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Remove(Delta):
    line_number: int

    def get_line_number(self) -> int:
        return self.line_number

    def get_type_name(self) -> str:
        return "Remove"

    def __hash__(self) -> int:
        return hash(self.line_number)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Remove):
            return False
        return self.line_number == other.line_number

    def __str__(self) -> str:
        return f"Remove({self.line_number})"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Replace(Delta):
    line_number: int
    value: str

    def get_type_name(self) -> str:
        return "Replace"

    def get_line_number(self) -> int:
        return self.line_number

    def __hash__(self) -> int:
        return hash((self.line_number, self.value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Replace):
            return False
        return self.line_number == other.line_number and \
            self.value == other.value

    def __str__(self) -> str:
        return f"Replace({self.line_number}, {self.value})"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Diff:
    deltas: List[Delta]

    def get_type_name(self) -> str:
        return "Diff"

    def __hash__(self) -> int:
        return hash(tuple(self.deltas))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Diff):
            return False
        return self.deltas == other.deltas

    def __str__(self) -> str:
        deltas = ", ".join(map(str, self.deltas))
        return f"Diff({deltas})"

    def __repr__(self) -> str:
        return str(self)


AST = Union[Diff, Delta]
