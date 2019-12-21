from dataclasses import dataclass
from typing import Union, List, Callable
from copy import deepcopy

from nl2prog.language import action

Tokenizer = Callable[[str], List[str]]


class AST:
    """
    Abstract syntax tree of the target language
    """

    def to_action_sequence(self, tokenizer: Tokenizer) \
            -> action.ActionSequence:
        """
        Return the action sequence corresponding to this AST

        Parameters
        ----------
        tokenizer: Tokenizer
            function to tokenize a string

        Returns
        -------
        action.ActionSequence
            The corresponding action sequence
        """
        pass

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

    def to_action_sequence(self, tokenizer: Tokenizer) \
            -> action.ActionSequence:
        """
        Return the action sequence corresponding to this AST

        Parameters
        ----------
        tokenizer: Tokenizer
            function to tokenize a string

        Returns
        -------
        action.ActionSequence
            The corresponding action sequence
        """
        def to_node_type(field: Field):
            if isinstance(field.value, list):
                return action.NodeType(field.type_name,
                                       action.NodeConstraint.Variadic)
            else:
                if isinstance(field.value, Leaf):
                    return action.NodeType(field.type_name,
                                           action.NodeConstraint.Token)
                else:
                    return action.NodeType(field.type_name,
                                           action.NodeConstraint.Node)
        children = list(map(lambda f: (f.name, to_node_type(f)), self.fields))

        seq = [action.ApplyRule(action.ExpandTreeRule(
            action.NodeType(self.type_name, action.NodeConstraint.Node),
            children))]
        for field in self.fields:
            if isinstance(field.value, list):
                for v in field.value:
                    seq.extend(v.to_action_sequence(tokenizer))
                seq.append(action.ApplyRule(action.CloseVariadicFieldRule()))
            else:
                seq.extend(field.value.to_action_sequence(tokenizer))
        return seq

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

    def to_action_sequence(self, tokenizer: Tokenizer) \
            -> action.ActionSequence:
        """
        Return the action sequence corresponding to this AST

        Parameters
        ----------
        tokenizer: Tokenizer
            function to tokenize a string

        Returns
        -------
        action.ActionSequence
            The corresponding action sequence
        """
        tokens: List[Union[str, action.CloseNode]] = tokenizer(str(self.value))
        tokens.append(action.CloseNode())
        return list(map(lambda x: action.GenerateToken(x), tokens))

    def clone(self):
        """
        Create and return the clone of this AST.

        Returns
        -------
        AST
            The cloned AST
        """
        return Leaf(self.type_name, deepcopy(self.value))
