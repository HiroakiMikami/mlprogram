import math
from functools import lru_cache
from typing import Callable, List, cast

import numpy as np

from mlprogram import logging
from mlprogram.languages import BatchedState
from mlprogram.languages import Interpreter as BaseInterpreter
from mlprogram.languages.csg import (
    AST,
    Circle,
    Difference,
    Rectangle,
    Reference,
    Rotation,
    Translation,
    Union,
)
from mlprogram.languages.csg.expander import Expander

logger = logging.Logger(__name__)


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

    def render(self, width: int, height: int, resolution: int = 1) -> np.ndarray:
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


class Interpreter(BaseInterpreter[AST, None, np.ndarray, str, None]):
    def __init__(self, width: int, height: int, resolution: int,
                 delete_used_reference: bool):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.delete_used_reference = delete_used_reference
        self._expander = Expander()

    def _render(self, shape: Shape) -> np.ndarray:
        return shape.render(self.width, self.height, self.resolution)

    def eval(self, code: AST, inputs: List[None]) -> List[np.ndarray]:
        return [self._render(self._cached_eval(code)) for _ in inputs]

    def create_state(self, inputs: List[None]) \
            -> BatchedState[AST, np.ndarray, str, None]:
        return BatchedState(
            type_environment={},
            environment={},
            history=[],
            context=inputs,
        )

    def execute(self, code: AST, state: BatchedState[AST, np.ndarray, str, None]) \
            -> BatchedState[AST, np.ndarray, str, None]:
        next = cast(BatchedState[AST, np.ndarray, str, None], state.clone())
        next.history.append(code)
        ref = Reference(len(next.history) - 1)
        next.type_environment[ref] = code.type_name()
        v = self._render(self._cached_eval(self._expander.unexpand(next.history)))
        value = [v for _ in state.context]
        next.environment[ref] = value

        if self.delete_used_reference:
            def _visit(code: AST):
                if isinstance(code, Circle) or isinstance(code, Rectangle):
                    return
                if isinstance(code, Rotation) or isinstance(code, Translation):
                    _visit(code.child)
                    return
                if isinstance(code, Union) or isinstance(code, Difference):
                    _visit(code.a)
                    _visit(code.b)
                    return
                if isinstance(code, Reference):
                    if code not in next.environment:
                        logger.warning(f"reference {code} is not found in environment")
                    else:
                        del next.environment[code]
                        del next.type_environment[code]
            _visit(code)
        return next

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
