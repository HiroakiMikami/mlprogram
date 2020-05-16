import numpy as np
from typing import cast
from mlprogram.interpreter import Environment, Interpreter as BaseInterpreter
from mlprogram.action.ast import AST, Node, Leaf


class Canvas:
    def __init__(self, width, height: int):
        self.width = width
        self.height = height
        self.canvas = np.zeros(height, width, dtype=np.bool)

    def render(self) -> str:
        retval = ""
        for y in self.canvas.shape[0]:
            for x in self.canvas.shape[1]:
                retval += "#" if self.canvas[y, x] else " "
            retval += "\n"
        return retval


class InvalidNodeTypeException(BaseException):
    def __init__(self, type_name: str):
        super(InvalidNodeTypeException).__init__(
            f"Invalid node type: {type_name}")


class Interpreter(BaseInterpreter):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def eval(self, env: Environment, ast: AST) -> Environment:
        assert isinstance(ast, Node)

        if ast in env:
            return env

        node = cast(Node, ast)
        if node.type_name == "Circle":
            # canvus = Canvas(self.width, self.height)
            r_node = node.fields[0].value
            assert isinstance(r_node, Leaf)
            r = int(r_node.value)
            print(r)
        elif node.type_name == "Rectangle":
            pass
        elif node.type_name == "Rotation":
            pass
        elif node.type_name == "Translation":
            pass
        elif node.type_name == "Union":
            pass
        elif node.type_name == "Difference":
            pass
        else:
            raise InvalidNodeTypeException(node.type_name)
