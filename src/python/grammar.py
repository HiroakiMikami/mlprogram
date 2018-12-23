import ast
from ast import *
import re
from enum import Enum
import typing
import inspect

from src.grammar import NodeType, Node, Rule, Sequence, ROOT, CLOSE_NODE

identifier = "identifier"
singleton = "singleton"
constant = "constant"


class ChildType(Enum):
    Standard = 0
    Variadic = 1
    Optional = 2


class Fields(typing.NamedTuple):
    fields: typing.List[typing.Tuple[str, ChildType]]


rules = {
    "Module":
    Fields([("body", ChildType.Variadic)]),
    "Expression":
    Fields([("body", ChildType.Standard)]),
    "FunctionDef":
    Fields([("name", ChildType.Standard), ("args", ChildType.Standard),
            ("body", ChildType.Variadic), ("decorator_list",
                                           ChildType.Variadic),
            ("returns", ChildType.Optional)]),
    "AsyncFunctionDef":
    Fields([("name", ChildType.Standard), ("args", ChildType.Standard),
            ("body", ChildType.Variadic), ("decorator_list",
                                           ChildType.Variadic),
            ("returns", ChildType.Optional)]),
    "ClassDef":
    Fields([("name", ChildType.Standard), ("bases", ChildType.Variadic),
            ("keywords", ChildType.Variadic), ("body", ChildType.Variadic),
            ("decorator_list", ChildType.Variadic)]),
    "Return":
    Fields([("value", ChildType.Optional)]),
    "Delete":
    Fields([("targets", ChildType.Variadic)]),
    "Assign":
    Fields([("targets", ChildType.Variadic), ("value", ChildType.Standard)]),
    "AugAssign":
    Fields([("target", ChildType.Standard), ("op", ChildType.Standard),
            ("value", ChildType.Standard)]),
    "AnnAssign":
    Fields([("target", ChildType.Standard), ("annotation", ChildType.Standard),
            ("value", ChildType.Optional), ("simple", ChildType.Standard)]),
    "For":
    Fields([("target", ChildType.Standard), ("iter", ChildType.Standard),
            ("body", ChildType.Variadic), ("orelse", ChildType.Variadic)]),
    "While":
    Fields([("test", ChildType.Standard), ("body", ChildType.Variadic),
            ("orelse", ChildType.Variadic)]),
    "If":
    Fields([("test", ChildType.Standard), ("body", ChildType.Variadic),
            ("orelse", ChildType.Variadic)]),
    "With":
    Fields([("items", ChildType.Variadic), ("body", ChildType.Variadic)]),
    "AsyncWith":
    Fields([("items", ChildType.Variadic), ("body", ChildType.Variadic)]),
    "Raise":
    Fields([("exc", ChildType.Optional), ("cause", ChildType.Optional)]),
    "Try":
    Fields([("body", ChildType.Variadic), ("handlers", ChildType.Variadic),
            ("orelse", ChildType.Variadic), ("finalbody",
                                             ChildType.Variadic)]),
    "Assert":
    Fields([("test", ChildType.Standard), ("msg", ChildType.Optional)]),
    "Import":
    Fields([("names", ChildType.Variadic)]),
    "ImportFrom":
    Fields([("module", ChildType.Optional), ("names", ChildType.Variadic),
            ("level", ChildType.Optional)]),
    "Global":
    Fields([("names", ChildType.Variadic)]),
    "Nonlocal":
    Fields([("names", ChildType.Variadic)]),
    "Expr":
    Fields([("value", ChildType.Standard)]),
    "Pass":
    Fields([]),
    "Break":
    Fields([]),
    "Continue":
    Fields([]),
    "BoolOp":
    Fields([("op", ChildType.Standard), ("values", ChildType.Variadic)]),
    "BinOp":
    Fields([("left", ChildType.Standard), ("op", ChildType.Standard),
            ("right", ChildType.Standard)]),
    "UnaryOp":
    Fields([("op", ChildType.Standard), ("operand", ChildType.Standard)]),
    "Lambda":
    Fields([("args", ChildType.Standard), ("body", ChildType.Standard)]),
    "IfExp":
    Fields([("test", ChildType.Standard), ("body", ChildType.Standard),
            ("orelse", ChildType.Standard)]),
    "Dict":
    Fields([("keys", ChildType.Variadic), ("values", ChildType.Variadic)]),
    "Set":
    Fields([("elts", ChildType.Variadic)]),
    "ListComp":
    Fields([("elt", ChildType.Standard), ("generators", ChildType.Variadic)]),
    "SetComp":
    Fields([("elt", ChildType.Standard), ("generators", ChildType.Variadic)]),
    "DictComp":
    Fields([("key", ChildType.Standard), ("value", ChildType.Standard),
            ("generators", ChildType.Variadic)]),
    "GeneratorExp":
    Fields([("elt", ChildType.Standard), ("generators", ChildType.Variadic)]),
    "Await":
    Fields([("value", ChildType.Standard)]),
    "Yield":
    Fields([("value", ChildType.Optional)]),
    "YieldFrom":
    Fields([("value", ChildType.Standard)]),
    "Compare":
    Fields([("left", ChildType.Standard), ("ops", ChildType.Variadic),
            ("comparators", ChildType.Variadic)]),
    "Call":
    Fields([("func", ChildType.Standard), ("args", ChildType.Variadic),
            ("keywords", ChildType.Variadic)]),
    "Num":
    Fields([("n", ChildType.Standard)]),
    "Str":
    Fields([("s", ChildType.Standard)]),
    "FormattedValue":
    Fields([("value", ChildType.Standard), ("conversion", ChildType.Optional),
            ("format_spec", ChildType.Optional)]),
    "JoinedStr":
    Fields([("values", ChildType.Variadic)]),
    "Bytes":
    Fields([("s", ChildType.Standard)]),
    "NameConstant":
    Fields([("value", ChildType.Standard)]),
    "Ellipsis":
    Fields([]),
    "Constant":
    Fields([("value", ChildType.Standard)]),

    # ctx is omitted
    "Attribute":
    Fields([("value", ChildType.Standard), ("attr", ChildType.Standard)]),
    "Subscript":
    Fields([("value", ChildType.Standard), ("slice", ChildType.Standard)]),
    "Starred":
    Fields([("value", ChildType.Standard)]),
    "Name":
    Fields([("id", ChildType.Standard)]),
    "List":
    Fields([("elts", ChildType.Variadic)]),
    "Tuple":
    Fields([("elts", ChildType.Variadic)]),
    "Slice":
    Fields([("lower", ChildType.Optional), ("upper", ChildType.Optional),
            ("step", ChildType.Optional)]),
    "ExtSlice":
    Fields([("dims", ChildType.Variadic)]),
    "Index":
    Fields([("value", ChildType.Standard)]),
    "And":
    Fields([]),
    "Or":
    Fields([]),
    "Add":
    Fields([]),
    "Sub":
    Fields([]),
    "Mult":
    Fields([]),
    "Div":
    Fields([]),
    "Mod":
    Fields([]),
    "Pow":
    Fields([]),
    "LShift":
    Fields([]),
    "RShift":
    Fields([]),
    "BitOr":
    Fields([]),
    "BitXor":
    Fields([]),
    "BitAnd":
    Fields([]),
    "FloorDiv":
    Fields([]),
    "Invert":
    Fields([]),
    "Not":
    Fields([]),
    "UAdd":
    Fields([]),
    "USub":
    Fields([]),
    "Eq":
    Fields([]),
    "NotEq":
    Fields([]),
    "Lt":
    Fields([]),
    "LtE":
    Fields([]),
    "Gt":
    Fields([]),
    "GtE":
    Fields([]),
    "Is":
    Fields([]),
    "IsNot":
    Fields([]),
    "In":
    Fields([]),
    "NotIn":
    Fields([]),
    "comprehension":
    Fields([("target", ChildType.Standard), ("iter", ChildType.Standard),
            ("ifs", ChildType.Variadic), ("is_async", ChildType.Standard)]),
    "ExceptHandler":
    Fields([("type", ChildType.Optional), ("name", ChildType.Optional),
            ("body", ChildType.Variadic)]),
    "arguments":
    Fields([("args", ChildType.Variadic), ("vararg", ChildType.Optional),
            ("kwonlyargs", ChildType.Variadic),
            ("kw_defaults", ChildType.Variadic), ("kwarg", ChildType.Optional),
            ("defaults", ChildType.Variadic)]),
    "arg":
    Fields([("arg", ChildType.Standard), ("annotation", ChildType.Optional)]),
    "keyword":
    Fields([("arg", ChildType.Optional), ("value", ChildType.Standard)]),
    "alias":
    Fields([("name", ChildType.Standard), ("asname", ChildType.Optional)]),
    "withitem":
    Fields([("context_expr", ChildType.Standard),
            ("optional_vars", ChildType.Optional)]),
}


def typename(x):
    return type(x).__name__


def is_builtin_type(node):
    if isinstance(node, ast.AST):
        return False
    elif isinstance(node, int):
        return True
    elif isinstance(node, float):
        return True
    elif isinstance(node, bool):
        return True
    elif isinstance(node, str):
        return True
    elif isinstance(node, bytes):
        return True
    elif isinstance(node, object):
        return True
    elif typename(node) == identifier:
        return True
    return False


def is_builtin_node_type(node_type: NodeType):
    if (node_type == ROOT):
        return False
    node = eval(node_type.type_name)

    if inspect.isclass(node):
        if issubclass(node, ast.AST):
            return False
        elif issubclass(node, int):
            return True
        elif issubclass(node, float):
            return True
        elif issubclass(node, bool):
            return True
        elif issubclass(node, str):
            return True
        elif issubclass(node, bytes):
            return True
        elif issubclass(node, object):
            return True
        return False
    elif node == identifier or node == singleton or node == constant:
        return True
    elif typename(node) == identifier or typename(
            node) == singleton or typename(node) == constant:
        return True
    return False


# TODO write unit test
def base_ast_type(node):
    base_types = set([
        ast.mod, ast.stmt, ast.expr, ast.expr_context, ast.slice, ast.boolop,
        ast.operator, ast.unaryop, ast.cmpop, ast.comprehension,
        ast.excepthandler, ast.arguments, ast.arg, ast.keyword, ast.alias,
        ast.withitem
    ])
    for base in base_types:
        if isinstance(node, base):
            return base
    return type(node)


# TODO write unit test
def convert_builtin_type(s, node_type: NodeType):
    nt = eval(node_type.type_name)
    if nt == object:
        try:
            s = int(s)
        except:
            try:
                s = float(s)
            except:
                pass
    elif not isinstance(nt, str):
        if nt == bytes:
            s = str(s).encode()
        elif nt == bool:
            if s == "True":
                return True
            else:
                return False
        else:
            s = nt(s)
    elif nt == singleton:
        if s is None:
            return s
        elif s == "None":
            return None
        elif s == "True":
            return True
        else:
            return False

    return s


def to_rule(node: ast.AST) -> Rule:
    parent = typename(node)
    children = []
    for chname, chval in ast.iter_fields(node):
        if chname == 'ctx':
            continue
        if chval is None or (isinstance(chval, list) and len(chval) == 0):
            continue

        is_list = isinstance(chval, list)
        if is_list:
            base_type = base_ast_type(chval[0])
        else:
            base_type = base_ast_type(chval)
        children.append(Node(chname, NodeType(base_type.__name__, is_list)))

    return Rule(NodeType(parent, False), tuple(children))


def to_sequence(node, split_camel_case=False):
    sequence = []

    def traverse(node, node_type: NodeType):
        if is_builtin_type(node):
            if isinstance(node, bytes):
                node_str = node.decode()
            else:
                node_str = "{}".format(node)

            if split_camel_case and re.search(
                    r"^[A-Z].*", node_str) and (not " " in node_str):
                # Camel Case
                words = re.findall(r"[A-Z][a-z]+", node_str)
                if "".join(words) == node_str:
                    for word in words:
                        sequence.append(word)
                else:
                    sequence.append(node_str)
            else:
                # Divide by space
                words = re.split(r"( +)", node_str)
                for word in words:
                    sequence.append(word)
            sequence.append(CLOSE_NODE)
        else:
            # node_type -> typename(node)
            this_type = NodeType(typename(node), False)
            sequence.append(Rule(node_type, (Node("-", this_type), )))

            # typename(node) -> childrens
            sequence.append(to_rule(node))
            for chname, chval in ast.iter_fields(node):
                if chname == 'ctx':
                    continue
                if chval is None or (isinstance(chval, list)
                                     and len(chval) == 0):
                    continue

                if isinstance(chval, list):
                    base_type_name = base_ast_type(chval[0]).__name__
                    # typename(chval)* -> typename(chval)...
                    cs = []
                    for i in range(len(chval)):
                        cs.append(
                            Node("val{}".format(i),
                                 NodeType(base_type_name, False)))
                    sequence.append(
                        Rule(NodeType(base_type_name, True), tuple(cs)))
                    for x in chval:
                        traverse(x, NodeType(base_type_name, False))
                else:
                    base_type_name = base_ast_type(chval).__name__
                    traverse(chval, NodeType(base_type_name, False))

    traverse(node, ROOT)
    return sequence


def to_ast(sequence: Sequence):
    def generate_subtree(node_type, index):
        assert (index < len(sequence))

        a = sequence[index]
        index = index
        index += 1
        if isinstance(a, str):
            assert (is_builtin_node_type(node_type))

            token = a
            # Generate Token
            if token != CLOSE_NODE:
                s, index = generate_subtree(node_type, index)
                return str(token + str(s)), index
            else:
                return "", index
        else:
            # ApplyRule
            rule = a

            assert (node_type == rule.parent)

            if len(rule.children) == 1 and rule.children[0].name == "-":
                return generate_subtree(rule.children[0].node_type, index)
            elif rule.parent.is_list:
                # List
                result = []
                for child in rule.children:
                    node_type = child.node_type
                    c, index = generate_subtree(node_type, index)
                    result.append(c)
                return result, index
            else:
                node = eval("{}()".format(rule.parent.type_name))
                if rule.parent.type_name in rules:
                    for name, child_type in rules[rule.parent.
                                                  type_name].fields:
                        if child_type == ChildType.Variadic:
                            setattr(node, name, [])
                        else:
                            setattr(node, name, None)
                for child in rule.children:
                    name = child.name
                    node_type = child.node_type
                    c, index = generate_subtree(node_type, index)
                    if is_builtin_node_type(node_type):
                        if isinstance(c, list):
                            c = list(map(lambda x: convert_builtin_type(x, node_type), c))
                            setattr(node, name, c)
                        else:
                            setattr(node, name,
                                    convert_builtin_type(c, node_type))
                    else:
                        setattr(node, name, c)
                return node, index

    tree, _ = generate_subtree(ROOT, 0)
    return tree
