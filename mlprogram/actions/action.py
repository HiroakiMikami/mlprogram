from dataclasses import dataclass
from mlprogram.asts import Root
from typing \
    import Tuple, Union, List, Any, TypeVar, Generic, Optional
from enum import Enum


class NodeConstraint(Enum):
    Token = 1
    Node = 2


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
    is_variadic: bool

    def __hash__(self) -> int:
        return hash(self.type_name) ^ hash(self.constraint) \
            ^ hash(self.is_variadic)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, NodeType):
            return self.type_name == rhs.type_name and \
                self.constraint == rhs.constraint and \
                self.is_variadic == rhs.is_variadic
        else:
            return False

    def __str__(self) -> str:
        if self.constraint == NodeConstraint.Node:
            value = str(self.type_name)
        elif self.constraint == NodeConstraint.Token:
            value = f"{self.type_name}(token)"
        if self.is_variadic:
            value = f"{value}*"
        return value


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

    def __hash__(self) -> int:
        return hash(self.rule)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, ApplyRule):
            return self.rule == rhs.rule
        else:
            return False


V = TypeVar("V")


@dataclass
class GenerateToken(Generic[V]):
    """
    The action to generate a token

    Attributes
    ----------
    token:
        The value (token) to be generated
    """
    token: V

    def __str__(self) -> str:
        return f"Generate {self.token}"

    def __hash__(self) -> int:
        return hash(self.token)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, GenerateToken):
            return self.token == rhs.token
        else:
            return False


Action = Union[ApplyRule, GenerateToken]
