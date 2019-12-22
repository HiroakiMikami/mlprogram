from dataclasses import dataclass
from typing import Union, List, Any
from copy import deepcopy


class AST:
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
        pass


@dataclass
class Field:
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
    type_name: str
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

    def __hash__(self):
        if isinstance(self.value, list):
            return hash(self.name) ^ hash(self.type_name) ^ \
                hash(tuple(self.value))
        else:
            return hash(self.name) ^ hash(self.type_name) ^ hash(self.value)

    def __eq__(self, rhs: Any):
        if isinstance(rhs, Field):
            return self.name == rhs.name and \
                self.type_name == rhs.type_name and self.value == rhs.value
        else:
            return False


@dataclass
class Node(AST):
    """
    The node of AST.

    Attributes
    ----------
    type_name: str
        The type of this node
    fields: List[Field]
        The list of fields
    """
    type_name: str
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

    def __hash__(self):
        return hash(self.type_name) ^ hash(tuple(self.fields))

    def __eq__(self, rhs: Any):
        if isinstance(rhs, Node):
            return self.type_name == rhs.type_name and \
                self.fields == rhs.fields
        else:
            return False


@dataclass
class Leaf(AST):
    """
    The leaf of AST

    Attributes
    ----------
    type_name: str
        The type of this leaf
    value: str
        The value represented by this leaf
    """
    type_name: str
    value: str

    def clone(self):
        """
        Create and return the clone of this AST.

        Returns
        -------
        AST
            The cloned AST
        """
        return Leaf(self.type_name, deepcopy(self.value))

    def __hash__(self):
        return hash(self.type_name) ^ hash(self.value)

    def __eq__(self, rhs: Any):
        if isinstance(rhs, Leaf):
            return self.type_name == rhs.type_name and \
                self.value == rhs.value
        else:
            return False
