import torch
from torchnlp.encoders import LabelEncoder
from typing import Any, List, Union
from dataclasses import dataclass

from nl2prog.language.action import NodeType, NodeConstraint
from nl2prog.language.action import ActionSequence, ActionOptions
from nl2prog.language.action import ApplyRule, GenerateToken
from nl2prog.language.action import Rule, ExpandTreeRule
from nl2prog.language.action import CloseNode, CloseVariadicFieldRule
from nl2prog.language.evaluator import Evaluator


def convert_node_type_to_key(node_type: NodeType):
    if node_type.constraint == NodeConstraint.Variadic:
        return NodeType(node_type.type_name, NodeConstraint.Node)
    else:
        return node_type


@dataclass
class ActionTensor:
    action: torch.LongTensor
    previous_action: torch.LongTensor


class Unknown:
    _instance = None

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, rhs: Any):
        return isinstance(rhs, Unknown)

    def __str__(self):
        return "<unknown>"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class ActionSequenceEncoder:
    def __init__(self,
                 rules: List[Rule], node_types: List[NodeType],
                 tokens: List[str], token_threshold: int,
                 options=ActionOptions(True, True)):
        reserved_labels = [Unknown()]
        if options.retain_vairadic_fields:
            reserved_labels.append(CloseVariadicFieldRule())
        self._rule_encoder = LabelEncoder(rules,
                                          reserved_labels=reserved_labels,
                                          unknown_index=0)
        self._node_type_encoder = LabelEncoder(list(
            map(convert_node_type_to_key, node_types)))
        reserved_labels = [Unknown()]
        if options.split_non_terminal:
            reserved_labels.append(CloseNode())
        self._token_encoder = LabelEncoder(tokens,
                                           min_occurrences=token_threshold,
                                           reserved_labels=reserved_labels,
                                           unknown_index=0)

    def decode(self, tensor: torch.LongTensor, query: List[str]) \
            -> Union[ActionSequence, None]:
        """
        Return the action sequence corresponding to the tensor

        Parameters
        ----------
        tensor: torch.LongTensor
            The encoded tensor with the shape of
            (len(action_sequence), 3). Each action will be encoded by the tuple
            of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).
            The padding value should be -1.
        query: List[str]

        Returns
        -------
        Union[ActionSequence, None]
            The action sequence corresponding to the tensor
            None if the action sequence cannot be generated.
        """

        retval: ActionSequence = []
        for i in range(tensor.shape[0]):
            if tensor[i, 0] > 0:
                # ApplyRule
                rule = self._rule_encoder.decode(tensor[i, 0])
                retval.append(ApplyRule(rule))
            elif tensor[i, 1] > 0:
                # GenerateToken
                token = self._token_encoder.decode(tensor[i, 1])
                retval.append(GenerateToken(token))
            elif tensor[i, 2] >= 0:
                # GenerateToken (Copy)
                index = int(tensor[i, 2].numpy())
                if index >= len(query):
                    return None
                token = query[index]
                retval.append(GenerateToken(token))
            else:
                return None

        return retval

    def encode_action(self, evaluator: Evaluator, query: List[str]) \
            -> Union[torch.Tensor, None]:
        """
        Return the tensor encoded the action sequence

        Parameters
        ----------
        evaluator: Evaluator
            The evaluator containing action sequence to be encoded
        query: List[str]

        Returns
        -------
        Union[torch.Tensor, None]
            The encoded tensor. The shape of tensor is
            (len(action_sequence) + 1, 4). Each action will be encoded by
            the tuple of (ID of the node types, ID of the applied rule,
            ID of the inserted token, the index of the word copied from
            the query. The padding value should be -1.
            None if the action sequence cannot be encoded.
        """
        action = torch.ones(len(evaluator.action_sequence) + 1, 4).long() * -1

        for i in range(len(evaluator.action_sequence)):
            a = evaluator.action_sequence[i]
            parent = evaluator.parent(i)
            if parent is not None:
                parent_action: ApplyRule = \
                    evaluator.action_sequence[parent.action]
                parent_rule: ExpandTreeRule = parent_action.rule
                action[i, 0] = self._node_type_encoder.encode(
                    convert_node_type_to_key(
                        parent_rule.children[parent.field][1]))

            if isinstance(a, ApplyRule):
                rule = a.rule
                action[i, 1] = self._rule_encoder.encode(rule)
            else:
                token = a.token
                encoded_token = int(self._token_encoder.encode(token).numpy())

                if encoded_token != 0:
                    action[i, 2] = encoded_token

                # Unknown token
                if token in query:
                    action[i, 3] = query.index(token)

                if encoded_token == 0 and token not in query:
                    return None

        head = evaluator.head
        length = len(evaluator.action_sequence)
        if head is not None:
            head_action: ApplyRule = \
                evaluator.action_sequence[head.action]
            head_rule: ExpandTreeRule = head_action.rule
            action[length, 0] = self._node_type_encoder.encode(
                convert_node_type_to_key(head_rule.children[head.field][1]))

        return action

    def encode_parent(self, evaluator) -> torch.Tensor:
        """
        Return the tensor encoded the action sequence

        Parameters
        ----------
        evaluator: Evaluator
            The evaluator containing action sequence to be encoded

        Returns
        -------
        torch.Tensor
            The encoded tensor. The shape of `action` tensor is
            (len(action_sequence) + 1, 4). Each action will be encoded by
            the tuple of (ID of the parent node types, ID of the
            parent-action's rule, the index of the parent action,
            the index of the field).
            The padding value should be -1.
        """
        parent_tensor = \
            torch.ones(len(evaluator.action_sequence) + 1, 4).long() * -1

        for i in range(len(evaluator.action_sequence)):
            parent = evaluator.parent(i)
            if parent is not None:
                parent_action: ApplyRule = \
                    evaluator.action_sequence[parent.action]
                parent_rule: ExpandTreeRule = parent_action.rule
                parent_tensor[i, 0] = self._node_type_encoder.encode(
                    convert_node_type_to_key(parent_rule.parent))
                parent_tensor[i, 1] = self._rule_encoder.encode(parent_rule)
                parent_tensor[i, 2] = parent.action
                parent_tensor[i, 3] = parent.field

        head = evaluator.head
        length = len(evaluator.action_sequence)
        if head is not None:
            head_action: ApplyRule = \
                evaluator.action_sequence[head.action]
            head_rule: ExpandTreeRule = head_action.rule
            parent_tensor[length, 0] = self._node_type_encoder.encode(
                convert_node_type_to_key(head_rule.parent))
            parent_tensor[length, 1] = self._rule_encoder.encode(head_rule)
            parent_tensor[length, 2] = head.action
            parent_tensor[length, 3] = head.field

        return parent_tensor

    @staticmethod
    def remove_variadic_node_types(node_types: List[NodeType]) \
            -> List[NodeType]:
        types = set([])
        retval = []
        for node_type in map(convert_node_type_to_key, node_types):
            if node_type not in types:
                retval.append(node_type)
                types.add(node_type)
        return retval
