from typing import Any, Dict
from mlprogram.interpreters import Reference as R


class AST:
    def type_name(self) -> str:
        raise NotImplementedError

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, AST):
            return self.type_name() == rhs.type_name() and \
                self.state_dict() == rhs.state_dict()
        return False

    def __hash__(self) -> int:
        return hash(self.type_name()) ^ hash(tuple(self.state_dict().items()))

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__str__()


class Rectangle(AST):
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def type_name(self) -> str:
        return "Rectangle"

    def state_dict(self) -> Dict[str, Any]:
        return {"w": self.w, "h": self.h}

    def __str__(self) -> str:
        return f"Rectangle(w={self.w},h={self.h})"


class Circle(AST):
    def __init__(self, r: int):
        self.r = r

    def type_name(self) -> str:
        return "Circle"

    def state_dict(self) -> Dict[str, Any]:
        return {"r": self.r}

    def __str__(self) -> str:
        return f"Circle(r={self.r})"


class Translation(AST):
    def __init__(self, x: int, y: int, child: AST):
        self.x = x
        self.y = y
        self.child = child

    def type_name(self) -> str:
        return "Translation"

    def state_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "child": self.child}

    def __str__(self) -> str:
        return f"Translation(x={self.x},y={self.y},child={self.child})"


class Rotation(AST):
    def __init__(self, theta_degree: int, child: AST):
        self.theta_degree = theta_degree
        self.child = child

    def type_name(self) -> str:
        return "Rotation"

    def state_dict(self) -> Dict[str, Any]:
        return {"theta_degree": self.theta_degree, "child": self.child}

    def __str__(self) -> str:
        return f"Rotation(theta={self.theta_degree},child={self.child})"


class Union(AST):
    def __init__(self, a: AST, b: AST):
        self.a = a
        self.b = b

    def type_name(self) -> str:
        return "Union"

    def state_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def __str__(self) -> str:
        return f"Union(a={self.a},b={self.b})"


class Difference(AST):
    def __init__(self, a: AST, b: AST):
        self.a = a
        self.b = b

    def type_name(self) -> str:
        return "Difference"

    def state_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def __str__(self) -> str:
        return f"Difference(a={self.a},b={self.b})"


class Reference(AST):
    def __init__(self, ref: R):
        self.ref = ref

    def type_name(self) -> str:
        return "Reference"

    def state_dict(self) -> Dict[str, Any]:
        return {"ref": self.ref}

    def __str__(self) -> str:
        return f"Reference(ref={self.ref})"
