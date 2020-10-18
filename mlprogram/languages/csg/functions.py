from typing import Union as U, List
from mlprogram.languages import Parser
from mlprogram.languages.csg import AST as csgAST
from mlprogram.languages.csg import Circle, Rectangle, Translation, Rotation
from mlprogram.languages.csg import Union, Difference, Reference
from mlprogram.languages.csg import Dataset
from mlprogram.encoders import Samples
from mlprogram.actions \
    import ActionSequence, ApplyRule, CloseVariadicFieldRule, Rule
from mlprogram.languages import Root


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


def get_samples(dataset: Dataset, parser: Parser[csgAST]) -> Samples:
    rules: List[Rule] = []
    node_types = []
    srule = set()
    sntype = set()
    tokens = [("size", x) for x in dataset.size_candidates]
    tokens.extend([("length", x) for x in dataset.length_candidates])
    tokens.extend([("degree", x) for x in dataset.degree_candidates])

    if dataset.reference:
        xs = [
            Circle(1), Rectangle(1, 2),
            Translation(1, 1, Reference(Circle(1))),
            Rotation(45, Reference(Circle(1))),
            Union(Reference(Circle(1)), Reference(Rectangle(1, 2))),
            Difference(Reference(Circle(1)), Reference(Rectangle(1, 2)))
        ]
    else:
        xs = [
            Circle(1), Rectangle(1, 2),
            Translation(1, 1, Circle(1)), Rotation(45, Circle(1)),
            Union(Circle(1), Circle(1)), Difference(Circle(1), Circle(1))
        ]

    for x in xs:
        ast = parser.parse(x)
        if ast is None:
            continue
        action_sequence = ActionSequence.create(ast)
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
    tokens = list(set(tokens))
    tokens.sort()

    return Samples(rules, node_types, tokens)
