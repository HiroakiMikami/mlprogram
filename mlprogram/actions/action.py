from dataclasses import dataclass
from mlprogram.asts import Root
from typing \
    import Tuple, Union, List, Any, TypeVar, Generic, Optional
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class NodeConstraint(Enum):
    Token = 1
    Node = 2
    Variadic = 3


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
    type_name: Optional[Union[str, Root]]
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


V = TypeVar("V")


@dataclass
class GenerateToken(Generic[V]):
    """
    The action to generate a token

    Attributes
    ----------
    token: Union[CloseNode, str]
        The value (token) to be generated
    """
    token: V

    def __str__(self) -> str:
        return f"Generate {self.token}"


Action = Union[ApplyRule, GenerateToken]
