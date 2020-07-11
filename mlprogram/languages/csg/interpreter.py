import numpy as np
from mlprogram.interpreters import Interpreter as BaseInterpreter
from mlprogram.languages.csg \
    import AST, Circle, Rectangle, Rotation, Translation, Union, Difference
from typing import Callable
import math
from functools import lru_cache


class Canvas:
    def __init__(self, width, height: int):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width), dtype=np.bool)

    def __str__(self) -> str:
        retval = ""
        for y in range(self.canvas.shape[0]):
            for x in range(self.canvas.shape[1]):
                retval += "#" if self.canvas[y, x] else " "
            retval += "\n"
        return retval


class Shape:
    def __init__(self, is_filled: Callable[[float, float], bool]):
        self.is_filled = is_filled

    def __call__(self, x: float, y: float) -> bool:
        return self.is_filled(x, y)

    def render(self, width: int, height: int) -> Canvas:
        canvas = Canvas(width, height)
        for y in range(height):
            for x in range(width):
                x_ = x - (width - 1) / 2
                y_ = y - (height - 1) / 2
                y_ *= -1
                if self(x_, y_):
                    canvas.canvas[y, x] = True
        return canvas


class InvalidNodeTypeException(BaseException):
    def __init__(self, type_name: str):
        super().__init__(f"Invalid node type: {type_name}")


class Interpreter(BaseInterpreter[AST, Shape]):
    def eval(self, code: AST) -> Shape:
        return self._cached_eval(code)

    @lru_cache(maxsize=100)
    def _cached_eval(self, code: AST) -> Shape:
        if isinstance(code, Circle):
            def circle(x, y):
                return x * x + y * y <= code.r * code.r
            return Shape(circle)
        elif isinstance(code, Rectangle):
            def rectangle(x, y):
                x = abs(x)
                y = abs(y)
                return x <= code.w / 2 and y <= code.h / 2
            return Shape(rectangle)
        elif isinstance(code, Translation):
            child = self.eval(code.child)

            def translate(x, y):
                x = x - code.x
                y = y - code.y
                return child(x, y)
            return Shape(translate)
        elif isinstance(code, Rotation):
            child = self.eval(code.child)

            def rotate(x, y):
                theta = math.radians(code.theta_degree)
                cos = math.cos(-theta)
                sin = math.sin(-theta)
                x_ = cos * x - sin * y
                y_ = sin * x + cos * y
                x, y = x_, y_
                return child(x, y)
            return Shape(rotate)
        elif isinstance(code, Union):
            a = self.eval(code.a)
            b = self.eval(code.b)

            def union(x, y):
                return a(x, y) or b(x, y)
            return Shape(union)
        elif isinstance(code, Difference):
            a = self.eval(code.a)
            b = self.eval(code.b)

            def difference(x, y):
                return not a(x, y) and b(x, y)
            return Shape(difference)
        raise InvalidNodeTypeException(code.type_name())
