import numpy as np
from mlprogram.interpreters import Reference as R
from mlprogram.interpreters import SequentialProgram
from mlprogram.interpreters import Interpreter as BaseInterpreter
from mlprogram.languages.csg \
    import AST, Circle, Rectangle, Rotation, Translation, Union, Difference, \
    Reference
from typing import Callable, Dict
import math
from functools import lru_cache


def show(canvas: np.array) -> str:
    retval = ""
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            retval += "#" if canvas[y, x] else " "
        retval += "\n"
    return retval


class Shape:
    def __init__(self, is_filled: Callable[[float, float], bool]):
        self.is_filled = is_filled

    def __call__(self, x: float, y: float) -> bool:
        return self.is_filled(x, y)

    def render(self, width: int, height: int, resolution: int = 1) -> np.array:
        canvas = np.zeros((height * resolution, width * resolution),
                          dtype=np.bool)
        for y in range(height * resolution):
            for x in range(width * resolution):
                x_ = (x - (width * resolution - 1) / 2) / resolution
                y_ = (y - (height * resolution - 1) / 2) / resolution
                y_ *= -1
                if self(x_, y_):
                    canvas[y, x] = True
        return canvas


class InvalidNodeTypeException(BaseException):
    def __init__(self, type_name: str):
        super().__init__(f"Invalid node type: {type_name}")


class Interpreter(BaseInterpreter[AST, Shape]):
    def __init__(self, width: int, height: int, resolution: int):
        self.width = width
        self.height = height
        self.resolution = resolution

    def eval(self, code: AST) -> np.array:
        return self._cached_eval(code).render(
            self.width, self.height, self.resolution)

    def eval_references(self, code: SequentialProgram[AST]) \
            -> Dict[R, np.array]:
        unref_code: Dict[R, AST] = {}
        values = {}
        for statement in code.statements:
            ref = statement.reference
            ast = statement.code
            unref_code[ref] = self._unreference(ast, unref_code)
            values[ref] = self.eval(unref_code[ref])
        return values

    def _unreference(self, code: AST, refs: Dict[R, AST]) -> AST:
        if isinstance(code, Circle):
            return code
        elif isinstance(code, Rectangle):
            return code
        elif isinstance(code, Translation):
            return Translation(code.x, code.y,
                               self._unreference(code.child, refs))
        elif isinstance(code, Rotation):
            return Rotation(code.theta_degree,
                            self._unreference(code.child, refs))
        elif isinstance(code, Union):
            return Union(self._unreference(code.a, refs),
                         self._unreference(code.b, refs))
        elif isinstance(code, Difference):
            return Difference(self._unreference(code.a, refs),
                              self._unreference(code.b, refs))
            a = self._cached_eval(code.a)
            b = self._cached_eval(code.b)

            def difference(x, y):
                return not a(x, y) and b(x, y)
            return Shape(difference)
        elif isinstance(code, Reference):
            return self._unreference(refs[code.ref], refs)
        raise InvalidNodeTypeException(code.type_name())

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
            child = self._cached_eval(code.child)

            def translate(x, y):
                x = x - code.x
                y = y - code.y
                return child(x, y)
            return Shape(translate)
        elif isinstance(code, Rotation):
            child = self._cached_eval(code.child)

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
            a = self._cached_eval(code.a)
            b = self._cached_eval(code.b)

            def union(x, y):
                return a(x, y) or b(x, y)
            return Shape(union)
        elif isinstance(code, Difference):
            a = self._cached_eval(code.a)
            b = self._cached_eval(code.b)

            def difference(x, y):
                return not a(x, y) and b(x, y)
            return Shape(difference)
        raise InvalidNodeTypeException(code.type_name())
