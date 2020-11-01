from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, TypeVar, Union

Kind = TypeVar("Kind")
V = TypeVar("V")


class Root:
    """
    The type of the root node
    """
    _instance = None

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, rhs: Any) -> bool:
        return isinstance(rhs, Root)

    def __str__(self) -> str:
        return "<Root>"

    def __repr__(self) -> str:
        return "<Root>"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class AST(Generic[Kind]):
    """
    Abstract syntax tree of the target language
    """

    def clone(self):
        """
        Create and return the clone of this AST.

        Returns
        -------
        AST
            The cloned AST
        """
        raise NotImplementedError

    def get_type_name(self) -> Optional[Union[Kind, Root]]:
        raise NotImplementedError


@dataclass
class Field(Generic[Kind]):
    """
    The field of the AST node.

    Attributes
    ----------
    name: str
        The name of this field
    type_name: str
        The type of this field
    value: Union[AST, List[AST]]
        The value of this field. It will be list of ASTs if this field is
        variadic.
    """
    name: str
    type_name: Union[Kind, Root]
    value: Union[AST, List[AST]]

    def clone(self):
        """
        Create and return the clone of this field.

        Returns
        -------
        Field
            The cloned field
        """
        if isinstance(self.value, list):
            return Field(self.name, self.type_name,
                         [v.clone() for v in self.value])
        else:
            return Field(self.name, self.type_name, self.value.clone())

    def __hash__(self) -> int:
        if isinstance(self.value, list):
            return hash(self.name) ^ hash(self.type_name) ^ \
                hash(tuple(self.value))
        else:
            return hash(self.name) ^ hash(self.type_name) ^ hash(self.value)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, Field):
            return self.name == rhs.name and \
                self.type_name == rhs.type_name and self.value == rhs.value
        else:
            return False


@dataclass
class Node(AST[Kind], Generic[Kind]):
    """
    The node of AST.

    Attributes
    ----------
    type_name: str
        The type of this node
    fields: List[Field]
        The list of fields
    """
    type_name: Optional[Union[Kind, Root]]
    fields: List[Field]

    def clone(self):
        """
        Create and return the clone of this AST.

        Returns
        -------
        AST
            The cloned AST
        """
        return Node(self.type_name,
                    [f.clone() for f in self.fields])

    def __hash__(self) -> int:
        return hash(self.type_name) ^ hash(tuple(self.fields))

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, Node):
            return self.type_name == rhs.type_name and \
                self.fields == rhs.fields
        else:
            return False

    def get_type_name(self) -> Optional[Union[Kind, Root]]:
        return self.type_name


@dataclass
class Leaf(AST[Kind], Generic[Kind, V]):
    """
    The leaf of AST

    Attributes
    ----------
    type_name: str
        The type of this leaf
    value: V
        The value represented by this leaf
    """
    type_name: Union[Kind, Root]
    value: V

    def clone(self):
        """
        Create and return the clone of this AST.

        Returns
        -------
        AST
            The cloned AST
        """
        return Leaf(self.type_name, deepcopy(self.value))

    def __hash__(self) -> int:
        return hash(self.type_name) ^ hash(self.value)

    def __eq__(self, rhs: Any) -> bool:
        if isinstance(rhs, Leaf):
            return self.type_name == rhs.type_name and \
                self.value == rhs.value
        else:
            return False

    def get_type_name(self) -> Optional[Union[Kind, Root]]:
        return self.type_name


class Sugar:
    @staticmethod
    def node(type_name: Optional[Union[Kind, Root]], **kwargs) -> Node:
        fields = [
            Field[Kind](name, kind, value)
            for name, (kind, value) in kwargs.items()
        ]
        return Node(type_name, fields)

    @staticmethod
    def leaf(type_name: Union[Kind, Root], value: V) -> Leaf[Kind, V]:
        return Leaf(type_name, value)
