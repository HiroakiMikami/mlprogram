from typing import List

from mlprogram.languages import Expander as BaseExpander
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


class Expander(BaseExpander[AST]):
    def expand(self, code: AST) -> List[AST]:
        retval: List[AST] = []

        def _visit(code: AST) -> Reference:
            if isinstance(code, Circle):
                retval.append(code)
            elif isinstance(code, Rectangle):
                retval.append(code)
            elif isinstance(code, Translation):
                retval.append(Translation(
                    x=code.x, y=code.y,
                    child=_visit(code.child)
                ))
            elif isinstance(code, Rotation):
                retval.append(Rotation(
                    theta_degree=code.theta_degree,
                    child=_visit(code.child)
                ))
            elif isinstance(code, Union):
                a = _visit(code.a)
                b = _visit(code.b)
                retval.append(Union(a=a, b=b))
            elif isinstance(code, Difference):
                a = _visit(code.a)
                b = _visit(code.b)
                retval.append(Difference(a=a, b=b))
            elif isinstance(code, Reference):
                retval.append(code)
                return code
            else:
                raise AssertionError(f"Invalid type: {code}")
            id = len(retval) - 1
            return Reference(id)
        _visit(code)
        return retval

    def unexpand(self, code: List[AST]) -> AST:
        ref_to_code = {
            Reference(i): c for i, c in enumerate(code)
        }

        def _visit(code: AST) -> AST:
            if isinstance(code, Circle):
                return code
            elif isinstance(code, Rectangle):
                return code
            elif isinstance(code, Translation):
                return Translation(code.x, code.y, _visit(code.child))
            elif isinstance(code, Rotation):
                return Rotation(code.theta_degree, _visit(code.child))
            elif isinstance(code, Union):
                return Union(_visit(code.a), _visit(code.b))
            elif isinstance(code, Difference):
                return Difference(_visit(code.a), _visit(code.b))
            elif isinstance(code, Reference):
                return _visit(ref_to_code[code])
            raise AssertionError(f"Invalid type: {code}")
        return _visit(code[-1])
