from typing import Union as U, Optional, Callable, List
from mlprogram.interpreters import Reference as R
from mlprogram.languages.csg import AST as csgAST
from mlprogram.languages.csg import Circle, Rectangle, Translation, Rotation
from mlprogram.languages.csg import Union, Difference, Reference
from mlprogram.languages.csg import Dataset
from mlprogram.encoders import Samples
from mlprogram.actions \
    import ActionSequence, ApplyRule, CloseVariadicFieldRule, Rule
from mlprogram.languages import AST, Node, Field, Leaf, Root
from mlprogram import logging

logger = logging.Logger(__name__)


class GetTokenType:
    def __call__(self, value: U[int, R]) -> Optional[str]:
        if isinstance(value, R):
            return "CSG"
        else:
            return "int"


class IsSubtype:
    def __call__(self, subtype: U[str, Root],
                 basetype: U[str, Root]) -> bool:
        if isinstance(basetype, Root):
            return True
        if basetype == "CSG":
            return subtype in set(["CSG", "Circle", "Rectangle", "Rotation",
                                   "Translation", "Union", "Difference"])
        if subtype == "int":
            return basetype in set(["size", "degree", "length"])
        return subtype == basetype


class ToAst:
    def __call__(self, code: csgAST) -> Optional[AST]:
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
            child = self(code.child)
            if child is None:
                return None
            else:
                return Node("Translation", [
                    Field("x", "length", Leaf("length", code.x)),
                    Field("y", "length", Leaf("length", code.y)),
                    Field("child", "CSG", child)
                ])
        elif isinstance(code, Rotation):
            child = self(code.child)
            if child is None:
                return None
            else:
                return Node("Rotation", [
                    Field("theta", "degree",
                          Leaf("degree", code.theta_degree)),
                    Field("child", "CSG", child)
                ])
        elif isinstance(code, Union):
            a, b = self(code.a), self(code.b)
            if a is None or b is None:
                return None
            else:
                return Node("Union", [
                    Field("a", "CSG", a),
                    Field("b", "CSG", b)
                ])
        elif isinstance(code, Difference):
            a, b = self(code.a), self(code.b)
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


class ToCsgAst:
    def __call__(self, code: AST) -> Optional[csgAST]:
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
                child = self(fields["child"])
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
                child = self(fields["child"])
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
                a, b = self(fields["a"]), self(fields["b"])
                if a is None or b is None:
                    return None
                else:
                    return Union(a, b)
            elif code.get_type_name() == "Difference":
                if not isinstance(fields["a"], AST):
                    return None
                if not isinstance(fields["b"], AST):
                    return None
                a, b = self(fields["a"]), self(fields["b"])
                if a is None or b is None:
                    return None
                else:
                    return Difference(a, b)
            return None
        elif isinstance(code, Leaf):
            return Reference(code.value)
        return None


def get_samples(dataset: Dataset,
                to_action_sequence: Callable[[csgAST],
                                             Optional[ActionSequence]]
                ) -> Samples:
    rules: List[Rule] = []
    node_types = []
    srule = set()
    sntype = set()
    tokens = dataset.size_candidates
    tokens.extend(dataset.length_candidates)
    tokens.extend(dataset.degree_candidates)

    if dataset.reference:
        xs = [
            Circle(1), Rectangle(1, 2),
            Translation(1, 1, Reference(R("0"))),
            Rotation(45, Reference(R("0"))),
            Union(Reference(R("0")), Reference(R("1"))),
            Difference(Reference(R("0")), Reference(R("1")))
        ]
    else:
        xs = [
            Circle(1), Rectangle(1, 2),
            Translation(1, 1, Circle(1)), Rotation(45, Circle(1)),
            Union(Circle(1), Circle(1)), Difference(Circle(1), Circle(1))
        ]

    for x in xs:
        action_sequence = to_action_sequence(x)
        if action_sequence is None:
            continue
        for action in action_sequence.action_sequence:
            if isinstance(action, ApplyRule):
                rule = action.rule
                if not isinstance(rule, CloseVariadicFieldRule):
                    if rule not in srule:
                        rules.append(rule)
                        srule.add(rule)
                    if rule.parent not in sntype:
                        node_types.append(rule.parent)
                        sntype.add(rule.parent)
                    for _, child in rule.children:
                        if child not in sntype:
                            node_types.append(child)
                            sntype.add(child)

    return Samples(list(rules), list(node_types), list(set(tokens)))
