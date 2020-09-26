from typing import List
from mlprogram.languages import AST, Node, Leaf, Field, Root
from mlprogram.languages import Token
from mlprogram.actions import Action, NodeType, NodeConstraint
from mlprogram.actions import ApplyRule, ExpandTreeRule
from mlprogram.actions import CloseVariadicFieldRule
from mlprogram.actions import GenerateToken
from mlprogram.actions import ActionSequence
from mlprogram import logging


logger = logging.Logger(__name__)


class InvalidNodeException(Exception):
    def __init__(self, node_type: str):
        super().__init__(f"Invalid type of node: {node_type}")


class AstToActionSequence:
    def __call__(self, node: AST) -> ActionSequence:
        """
        Return the action sequence corresponding to this AST

        Parameters
        ----------
        node: AST

        Returns
        -------
        actionSequence
            The corresponding action sequence
        """
        def to_sequence(node: AST) -> List[Action]:
            if isinstance(node, Node):
                def to_node_type(field: Field) -> NodeType:
                    if isinstance(field.value, list):
                        if len(field.value) > 0 and \
                                isinstance(field.value[0], Leaf):
                            return NodeType(field.type_name,
                                            NodeConstraint.Token, True)
                        else:
                            return NodeType(field.type_name,
                                            NodeConstraint.Node, True)
                    else:
                        if isinstance(field.value, Leaf):
                            return NodeType(field.type_name,
                                            NodeConstraint.Token,
                                            False)
                        else:
                            return NodeType(field.type_name,
                                            NodeConstraint.Node, False)
                children = list(
                    map(lambda f: (f.name, to_node_type(f)), node.fields))

                seq: List[Action] = [ApplyRule(ExpandTreeRule(
                    NodeType(node.type_name, NodeConstraint.Node, False),
                    children))]
                for field in node.fields:
                    if isinstance(field.value, list):
                        for v in field.value:
                            seq.extend(to_sequence(v))
                        seq.append(ApplyRule(CloseVariadicFieldRule()))
                    else:
                        seq.extend(to_sequence(field.value))
                return seq
            elif isinstance(node, Leaf):
                node_type = node.get_type_name()
                assert not isinstance(node_type, Root)

                return [GenerateToken(Token(node_type, node.value,
                                            node.value))]
            else:
                logger.critical(f"Invalid type of node: {type(node)}")
                raise InvalidNodeException(str(type(node)))
        action_sequence = ActionSequence()
        node = Node(None, [Field("root", Root(), node)])
        for action in to_sequence(node):
            action_sequence.eval(action)
        return action_sequence
