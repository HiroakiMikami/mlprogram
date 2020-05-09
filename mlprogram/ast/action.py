from dataclasses import dataclass
from mlprogram.ast.ast import AST, Node, Leaf, Field, Root
from typing import Tuple, Union, List, Any, Callable, Optional, Sequence
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class NodeConstraint(Enum):
    Token = 1
    Node = 2
    Variadic = 3


@dataclass
class ActionOptions:
    retain_variadic_fields: bool
    split_non_terminal: bool


@dataclass
class NodeType:
    """
    The type of the AST node

    Attributes
    ----------
    type_name: Union[str, Root]
    constraint: NodeConstraint
        It represents the constraint of this type
    """
    type_name: Union[str, Root]
    constraint: NodeConstraint

    def __hash__(self) -> int:
        return hash(self.type_name) ^ hash(self.constraint)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, NodeType):
            return self.type_name == rhs.type_name and \
                self.constraint == rhs.constraint
        else:
            return False

    def __str__(self) -> str:
        if self.constraint == NodeConstraint.Variadic:
            return f"{self.type_name}*"
        elif self.constraint == NodeConstraint.Token:
            return f"{self.type_name}(token)"
        else:
            return str(self.type_name)


@dataclass
class ExpandTreeRule:
    """
    Rule that expands AST

    Attributes
    ----------
    parent: NodeType
        The current node type
    children: List[Tuple[str, NodeType]]
        The node types of the fields
    """
    parent: NodeType
    children: List[Tuple[str, NodeType]]

    def __hash__(self) -> int:
        return hash(self.parent) ^ hash(tuple(self.children))

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, ExpandTreeRule):
            return self.parent == rhs.parent and self.children == rhs.children
        else:
            return False

    def __str__(self) -> str:
        children = ", ".join(
            map(lambda x: f"{x[0]}: {x[1]}", self.children))
        return f"{self.parent} -> [{children}]"


class CloseVariadicFieldRule:
    """
    The rule that closes the variadic field
    """
    _instance = None

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, rhs: Any) -> bool:
        return isinstance(rhs, CloseVariadicFieldRule)

    def __str__(self) -> str:
        return "<close variadic field>"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


Rule = Union[ExpandTreeRule, CloseVariadicFieldRule]


@dataclass
class ApplyRule:
    """
    The action to apply a rule

    Attributes
    ----------
    rule: Rule
    """
    rule: Rule

    def __str__(self) -> str:
        return f"Apply ({self.rule})"


class CloseNode:
    """
    The value to stop generating tokens
    """
    _instance = None

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, rhs: Any) -> bool:
        return isinstance(rhs, CloseNode)

    def __str__(self) -> str:
        return "<CLOSE_NODE>"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


@dataclass
class GenerateToken:
    """
    The action to generate a token

    Attributes
    ----------
    token: Union[CloseNode, str]
        The value (token) to be generated
    """
    token: Union[CloseNode, str]

    def __str__(self) -> str:
        return f"Generate {self.token}"


Action = Union[ApplyRule, GenerateToken]


@dataclass
class ActionSequence:
    sequence: List[Action]
    options: ActionOptions


Tokenizer = Callable[[str], Sequence[str]]


def ast_to_action_sequence(node: AST,
                           options: ActionOptions = ActionOptions(True, True),
                           tokenizer: Optional[Tokenizer] = None) \
        -> ActionSequence:
    """
    Return the action sequence corresponding to this AST

    Parameters
    ----------
    node: AST
    options: ActionOptions
    tokenizer: Optional[Tokenizer]
        function to tokenize a string.
        This is required if the options.split_non_terminal is True.

    Returns
    -------
    action.ActionSequence
        The corresponding action sequence
    """
    def to_sequence(node: AST) -> List[Action]:
        if isinstance(node, Node):
            def to_node_type(field: Field) -> NodeType:
                if isinstance(field.value, list):
                    return NodeType(field.type_name,
                                    NodeConstraint.Variadic)
                else:
                    if isinstance(field.value, Leaf):
                        return NodeType(field.type_name,
                                        NodeConstraint.Token)
                    else:
                        return NodeType(field.type_name,
                                        NodeConstraint.Node)
            children = list(
                map(lambda f: (f.name, to_node_type(f)), node.fields))

            seq: List[Action] = [ApplyRule(ExpandTreeRule(
                NodeType(node.type_name, NodeConstraint.Node),
                children))]
            for field in node.fields:
                if isinstance(field.value, list):
                    if not options.retain_variadic_fields:
                        elem_type_name = to_node_type(field).type_name
                        elem = NodeType(elem_type_name, NodeConstraint.Node)
                        seq.append(ApplyRule(ExpandTreeRule(
                            NodeType(elem_type_name, NodeConstraint.Variadic),
                            [(str(i), elem) for i in range(len(field.value))]
                        )))
                    for v in field.value:
                        seq.extend(to_sequence(v))
                    if options.retain_variadic_fields:
                        seq.append(ApplyRule(CloseVariadicFieldRule()))
                else:
                    seq.extend(to_sequence(field.value))
            return seq
        elif isinstance(node, Leaf):
            if options.split_non_terminal:
                assert tokenizer is not None
                tokens: List[Union[str, CloseNode]] = \
                    list(tokenizer(str(node.value)))
                tokens.append(CloseNode())
                return list(map(lambda x: GenerateToken(x), tokens))
            else:
                return [GenerateToken(node.value)]
        else:
            logger.warn(f"Invalid type of node: {type(node)}")
            return []
    return ActionSequence(
        to_sequence(Node(Root(), [Field("root", Root(), node)])),
        options)


def code_to_action_sequence(
    code: str, parse: Callable[[str], AST],
    options: ActionOptions,
    tokenize: Optional[Callable[[str], List[str]]] = None) \
        -> Optional[ActionSequence]:
    ast = parse(code)
    if ast is None:
        return None
    return ast_to_action_sequence(ast, tokenizer=tokenize, options=options)