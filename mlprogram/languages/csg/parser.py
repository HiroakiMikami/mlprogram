from typing import Optional
from mlprogram.languages.csg import AST as csgAST
from mlprogram.languages.csg import Circle, Rectangle, Translation, Rotation
from mlprogram.languages.csg import Union, Difference, Reference
from mlprogram.languages import AST, Node, Field, Leaf
from mlprogram.languages import Parser as BaseParser
from mlprogram import logging

logger = logging.Logger(__name__)


class Parser(BaseParser[csgAST]):
    def parse(self, code: csgAST) -> Optional[AST]:
        if isinstance(code, Circle):
            return Node("Circle", [
                Field("r", "size", Leaf("size", code.r))
            ])
        elif isinstance(code, Rectangle):
            return Node("Rectangle", [
                Field("w", "size", Leaf("size", code.w)),
                Field("h", "size", Leaf("size", code.h))
            ])
        elif isinstance(code, Translation):
            child = self.parse(code.child)
            if child is None:
                return None
            else:
                return Node("Translation", [
                    Field("x", "length", Leaf("length", code.x)),
                    Field("y", "length", Leaf("length", code.y)),
                    Field("child", "CSG", child)
                ])
        elif isinstance(code, Rotation):
            child = self.parse(code.child)
            if child is None:
                return None
            else:
                return Node("Rotation", [
                    Field("theta", "degree",
                          Leaf("degree", code.theta_degree)),
                    Field("child", "CSG", child)
                ])
        elif isinstance(code, Union):
            a, b = self.parse(code.a), self.parse(code.b)
            if a is None or b is None:
                return None
            else:
                return Node("Union", [
                    Field("a", "CSG", a),
                    Field("b", "CSG", b)
                ])
        elif isinstance(code, Difference):
            a, b = self.parse(code.a), self.parse(code.b)
            if a is None or b is None:
                return None
            else:
                return Node("Difference", [
                    Field("a", "CSG", a),
                    Field("b", "CSG", b)
                ])
        elif isinstance(code, Reference):
            return Leaf("CSG", code.ref)
        logger.warning(f"Invalid node type {code.type_name()}")
        # TODO throw exception
        return None

    def unparse(self, code: AST) -> Optional[csgAST]:
        if isinstance(code, Node):
            fields = {field.name: field.value for field in code.fields}
            if code.get_type_name() == "Circle":
                if isinstance(fields["r"], Leaf):
                    return Circle(fields["r"].value)
                else:
                    return None
            elif code.get_type_name() == "Rectangle":
                if isinstance(fields["w"], Leaf) and \
                        isinstance(fields["h"], Leaf):
                    return Rectangle(fields["w"].value, fields["h"].value)
                else:
                    return None
            elif code.get_type_name() == "Translation":
                if not isinstance(fields["child"], AST):
                    return None
                child = self.unparse(fields["child"])
                if child is None:
                    return None
                else:
                    if isinstance(fields["x"], Leaf) and \
                            isinstance(fields["y"], Leaf):
                        return Translation(
                            fields["x"].value, fields["y"].value,
                            child
                        )
                    else:
                        return None
            elif code.get_type_name() == "Rotation":
                if not isinstance(fields["child"], AST):
                    return None
                child = self.unparse(fields["child"])
                if child is None:
                    return None
                else:
                    if isinstance(fields["theta"], Leaf):
                        return Rotation(
                            fields["theta"].value,
                            child,
                        )
                    else:
                        return None
            elif code.get_type_name() == "Union":
                if not isinstance(fields["a"], AST):
                    return None
                if not isinstance(fields["b"], AST):
                    return None
                a, b = self.unparse(fields["a"]), self.unparse(fields["b"])
                if a is None or b is None:
                    return None
                else:
                    return Union(a, b)
            elif code.get_type_name() == "Difference":
                if not isinstance(fields["a"], AST):
                    return None
                if not isinstance(fields["b"], AST):
                    return None
                a, b = self.unparse(fields["a"]), self.unparse(fields["b"])
                if a is None or b is None:
                    return None
                else:
                    return Difference(a, b)
            return None
        elif isinstance(code, Leaf):
            return Reference(code.value)
        return None
