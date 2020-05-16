import numpy as np
from mlprogram.interpreter import Environment, Interpreter as BaseInterpreter
from mlprogram.csg \
    import AST, Circle, Rectangle, Rotation, Translation, Union, Difference


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

    def eval(self, env: Environment, code: AST) -> Environment:
        if code in env:
            return env

        if isinstance(code, Circle):
            # canvus = Canvas(self.width, self.height)
            pass
        elif isinstance(code, Rectangle):
            pass
        elif isinstance(code, Rotation):
            pass
        elif isinstance(code, Translation):
            pass
        elif isinstance(code, Union):
            pass
        elif isinstance(code, Difference):
            pass
        else:
            raise InvalidNodeTypeException(code.type_name())
