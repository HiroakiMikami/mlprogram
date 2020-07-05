from typing import Callable, Sequence, Optional, List
import logging
from mlprogram.asts import AST, Node, Leaf, Field, Root
from mlprogram.actions import ActionOptions
from mlprogram.actions import Action, NodeType, NodeConstraint
from mlprogram.actions import ApplyRule, ExpandTreeRule
from mlprogram.actions import CloseVariadicFieldRule
from mlprogram.actions import GenerateToken
from mlprogram.actions import ActionSequence


logger = logging.getLogger(__name__)


class AstToSingleActionSequence:
    def __init__(self, options: ActionOptions = ActionOptions(True, True),
                 tokenize: Optional[Callable[[str], Sequence[str]]] = None):
        """
        Return the action sequence corresponding to this AST

        Parameters
        ----------
        options:
        tokenize:
            function to tokenize a string.
            This is required if the options.split_non_terminal is True.
        """
        if options.split_non_terminal:
            assert tokenize is not None

        self.options = options
        self.tokenize = tokenize

    def __call__(self, node: AST) -> ActionSequence:
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
        actionSequence
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
                        if not self.options.retain_variadic_fields:
                            elem_type_name = to_node_type(field).type_name
                            elem = NodeType(elem_type_name,
                                            NodeConstraint.Node)
                            seq.append(ApplyRule(ExpandTreeRule(
                                NodeType(elem_type_name,
                                         NodeConstraint.Variadic),
                                [(str(i), elem)
                                 for i in range(len(field.value))]
                            )))
                        for v in field.value:
                            seq.extend(to_sequence(v))
                        if self.options.retain_variadic_fields:
                            seq.append(ApplyRule(CloseVariadicFieldRule()))
                    else:
                        seq.extend(to_sequence(field.value))
                return seq
            elif isinstance(node, Leaf):
                if self.options.split_non_terminal:
                    gen_tokens: List[Action] = []
                    if isinstance(node.value, str):
                        assert self.tokenize is not None
                        for token in self.tokenize(node.value):
                            gen_tokens.append(GenerateToken[str](token))
                    else:
                        gen_tokens.append(GenerateToken(node.value))
                    gen_tokens.append(ApplyRule(CloseVariadicFieldRule()))
                    return gen_tokens
                else:
                    return [GenerateToken(node.value)]
            else:
                logger.warn(f"Invalid type of node: {type(node)}")
                return []
        action_sequence = ActionSequence(self.options)
        node = Node(None, [Field("root", Root(), node)])
        for action in to_sequence(node):
            action_sequence.eval(action)
        return action_sequence
