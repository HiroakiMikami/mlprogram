import numpy as np
from mlprogram.interpreter import Environment, Interpreter as BaseInterpreter
from mlprogram.language.csg \
    import AST, Circle, Rectangle, Rotation, Translation, Union, Difference
from typing import Callable
import math


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


class Interpreter(BaseInterpreter):
    def eval(self, env: Environment, code: AST) -> Environment:
        if code in env:
            return env

        if isinstance(code, Circle):
            def circle(x, y):
                return x * x + y * y <= code.r * code.r
            env[code] = Shape(circle)
        elif isinstance(code, Rectangle):
            def rectangle(x, y):
                x = abs(x)
                y = abs(y)
                return x <= code.w / 2 and y <= code.h / 2
            env[code] = Shape(rectangle)
        elif isinstance(code, Translation):
            env = self.eval(env, code.child)

            def translate(x, y):
                x = x - code.x
                y = y - code.y
                return env[code.child](x, y)
            env[code] = Shape(translate)
        elif isinstance(code, Rotation):
            env = self.eval(env, code.child)

            def rotate(x, y):
                theta = math.radians(code.theta_degree)
                cos = math.cos(-theta)
                sin = math.sin(-theta)
                x_ = cos * x - sin * y
                y_ = sin * x + cos * y
                x, y = x_, y_
                return env[code.child](x, y)
            env[code] = Shape(rotate)
        elif isinstance(code, Union):
            env = self.eval(env, code.a)
            env = self.eval(env, code.b)

            def union(x, y):
                return env[code.a](x, y) or env[code.b](x, y)
            env[code] = Shape(union)
        elif isinstance(code, Difference):
            env = self.eval(env, code.a)
            env = self.eval(env, code.b)

            def difference(x, y):
                return not env[code.a](x, y) and env[code.b](x, y)
            env[code] = Shape(difference)
        else:
            raise InvalidNodeTypeException(code.type_name())
        return env
