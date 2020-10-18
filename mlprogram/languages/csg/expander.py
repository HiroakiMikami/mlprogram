from typing import List
from mlprogram.languages.csg import AST
from mlprogram.languages.csg import Circle
from mlprogram.languages.csg import Rectangle
from mlprogram.languages.csg import Translation
from mlprogram.languages.csg import Rotation
from mlprogram.languages.csg import Union
from mlprogram.languages.csg import Difference
from mlprogram.languages.csg import Reference
from mlprogram.languages import Expander as BaseExpander


class Expander(BaseExpander[AST]):
    def expand(self, code: AST) -> List[AST]:
        retval = []

        def _visit(code: AST):
            if isinstance(code, Circle):
                pass
            elif isinstance(code, Rectangle):
                pass
            elif isinstance(code, Translation):
                _visit(code.child)
            elif isinstance(code, Rotation):
                _visit(code.child)
            elif isinstance(code, Union):
                _visit(code.a)
                _visit(code.b)
            elif isinstance(code, Difference):
                _visit(code.a)
                _visit(code.b)
            elif isinstance(code, Reference):
                _visit(code.ref)
                retval.append(code.ref)
        _visit(code)
        retval.append(code)
        return retval

    def unexpand(self, code: List[AST]) -> AST:
        def _visit(code: AST) -> AST:
            if isinstance(code, Circle):
                return code
            elif isinstance(code, Rectangle):
                return code
            elif isinstance(code, Translation):
                return Translation(code.x, code.y,
                                   _visit(code.child))
            elif isinstance(code, Rotation):
                return Rotation(code.theta_degree,
                                _visit(code.child))
            elif isinstance(code, Union):
                return Union(_visit(code.a),
                             _visit(code.b))
            elif isinstance(code, Difference):
                return Difference(_visit(code.a),
                                  _visit(code.b))
            elif isinstance(code, Reference):
                return Reference(_visit(code.ref))
            raise AssertionError()
        return _visit(code[-1])
