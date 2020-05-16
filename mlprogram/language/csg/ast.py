from dataclasses import dataclass


class AST:
    def type_name(self) -> str:
        raise NotImplementedError


@dataclass
class Rectangle(AST):
    w: int
    h: int

    def type_name(self) -> str:
        return "Rectangle"


@dataclass
class Circle(AST):
    r: int

    def type_name(self) -> str:
        return "Circle"


@dataclass
class Translation(AST):
    x: int
    y: int
    child: AST

    def type_name(self) -> str:
        return "Translation"


@dataclass
class Rotation(AST):
    theta_degree: int
    child: AST

    def type_name(self) -> str:
        return "Rotation"


@dataclass
class Union(AST):
    a: AST
    b: AST

    def type_name(self) -> str:
        return "Union"


@dataclass
class Difference(AST):
    a: AST
    b: AST

    def type_name(self) -> str:
        return "Difference"
