import torch
from torchnlp.encoders import LabelEncoder
from typing import Any, List, Optional, Union, cast, Generic, TypeVar
from typing import Dict
from dataclasses import dataclass

from mlprogram.languages import Token
from mlprogram.actions import NodeType
from mlprogram.actions import ApplyRule, GenerateToken
from mlprogram.actions import Rule, ExpandTreeRule
from mlprogram.actions import CloseVariadicFieldRule
from mlprogram.actions import ActionSequence

V = TypeVar("V")


@dataclass
class ActionTensor:
    action: torch.LongTensor
    previous_action: torch.LongTensor


class Unknown:
    _instance = None

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, rhs: Any) -> bool:
        return isinstance(rhs, Unknown)

    def __str__(self) -> str:
        return "<unknown>"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


@dataclass
class Samples(Generic[V]):
    rules: List[Rule]
    node_types: List[NodeType]
    tokens: List[Token[str, V]]


class ActionSequenceEncoder:
    def __init__(self, samples: Samples, token_threshold: int):
        reserved_labels: List[Union[Unknown,
                                    CloseVariadicFieldRule]] = [Unknown()]
        reserved_labels.append(CloseVariadicFieldRule())
        self._rule_encoder = LabelEncoder(samples.rules,
                                          reserved_labels=reserved_labels,
                                          unknown_index=0)
        self._node_type_encoder = LabelEncoder(samples.node_types)
        reserved_labels = [Unknown()]
        self._token_encoder = LabelEncoder(samples.tokens,
                                           min_occurrences=token_threshold,
                                           reserved_labels=reserved_labels,
                                           unknown_index=0)
        self.raw_value_to_idx: Dict[str, List[int]] = {}
        for token in self._token_encoder.vocab:
            if not isinstance(token, Token):
                continue
            idx = self._token_encoder.encode(token)
            if token.raw_value not in self.raw_value_to_idx:
                self.raw_value_to_idx[token.raw_value] = []
            self.raw_value_to_idx[token.raw_value].append(idx)

    def decode(self, tensor: torch.LongTensor, query: List[str]) \
            -> Optional[ActionSequence]:
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
        Optional[action_sequence]
            The action sequence corresponding to the tensor
            None if the action sequence cannot be generated.
        """

        retval = ActionSequence()
        for i in range(tensor.shape[0]):
            if tensor[i, 0] > 0:
                # ApplyRule
                rule = self._rule_encoder.decode(tensor[i, 0])
                retval.eval(ApplyRule(rule))
            elif tensor[i, 1] > 0:
                # GenerateToken
                token = self._token_encoder.decode(tensor[i, 1])
                retval.eval(GenerateToken(token))
            elif tensor[i, 2] >= 0:
                # GenerateToken (Copy)
                index = int(tensor[i, 2].numpy())
                if index >= len(query):
                    return None
                token = Token(None, query[index], query[index])
                retval.eval(GenerateToken(token))
            else:
                return None

        return retval

    def encode_action(self,
                      action_sequence: ActionSequence, query: List[str]) \
            -> Optional[torch.Tensor]:
        """
        Return the tensor encoded the action sequence

        Parameters
        ----------
        action_sequence: action_sequence
            The action_sequence containing action sequence to be encoded
        query: List[str]

        Returns
        -------
        Optional[torch.Tensor]
            The encoded tensor. The shape of tensor is
            (len(action_sequence) + 1, 4). Each action will be encoded by
            the tuple of (ID of the node types, ID of the applied rule,
            ID of the inserted token, the index of the word copied from
            the query. The padding value should be -1.
            None if the action sequence cannot be encoded.
        """
        action = \
            torch.ones(len(action_sequence.action_sequence) + 1, 4).long() \
            * -1
        for i in range(len(action_sequence.action_sequence)):
            a = action_sequence.action_sequence[i]
            parent = action_sequence.parent(i)
            if parent is not None:
                parent_action = \
                    cast(ApplyRule,
                         action_sequence.action_sequence[parent.action])
                parent_rule = cast(ExpandTreeRule, parent_action.rule)
                action[i, 0] = self._node_type_encoder.encode(
                    parent_rule.children[parent.field][1])

            if isinstance(a, ApplyRule):
                rule = a.rule
                action[i, 1] = self._rule_encoder.encode(rule)
            else:
                token = a.token
                encoded_token = int(self._token_encoder.encode(token).numpy())

                if encoded_token != 0:
                    action[i, 2] = encoded_token

                # Unknown token
                if token.raw_value in query:
                    # TODO
                    action[i, 3] = query.index(cast(str, token.raw_value))

                if encoded_token == 0 and token.raw_value not in query:
                    return None

        head = action_sequence.head
        length = len(action_sequence.action_sequence)
        if head is not None:
            head_action = \
                cast(ApplyRule,
                     action_sequence.action_sequence[head.action])
            head_rule = cast(ExpandTreeRule, head_action.rule)
            action[length, 0] = self._node_type_encoder.encode(
                head_rule.children[head.field][1])

        return action

    def encode_raw_value(self, text: str) -> List[int]:
        if text in self.raw_value_to_idx:
            return self.raw_value_to_idx[text]
        else:
            return [self._token_encoder.encode(Unknown()).item()]

    def batch_encode_raw_value(self, texts: List[str]) -> List[List[int]]:
        return [
            self.encode_raw_value(text)
            for text in texts
        ]

    def encode_parent(self, action_sequence) -> torch.Tensor:
        """
        Return the tensor encoded the action sequence

        Parameters
        ----------
        action_sequence: action_sequence
            The action_sequence containing action sequence to be encoded

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
            torch.ones(len(action_sequence.action_sequence) + 1, 4).long() \
            * -1

        for i in range(len(action_sequence.action_sequence)):
            parent = action_sequence.parent(i)
            if parent is not None:
                parent_action = \
                    cast(ApplyRule,
                         action_sequence.action_sequence[parent.action])
                parent_rule = cast(ExpandTreeRule, parent_action.rule)
                parent_tensor[i, 0] = \
                    self._node_type_encoder.encode(parent_rule.parent)
                parent_tensor[i, 1] = self._rule_encoder.encode(parent_rule)
                parent_tensor[i, 2] = parent.action
                parent_tensor[i, 3] = parent.field

        head = action_sequence.head
        length = len(action_sequence.action_sequence)
        if head is not None:
            head_action = \
                cast(ApplyRule,
                     action_sequence.action_sequence[head.action])
            head_rule = cast(ExpandTreeRule, head_action.rule)
            parent_tensor[length, 0] = \
                self._node_type_encoder.encode(head_rule.parent)
            parent_tensor[length, 1] = self._rule_encoder.encode(head_rule)
            parent_tensor[length, 2] = head.action
            parent_tensor[length, 3] = head.field

        return parent_tensor

    def encode_tree(self, action_sequence: ActionSequence) \
            -> Union[torch.Tensor, torch.Tensor]:
        """
        Return the tensor adjacency matrix of the action sequence

        Parameters
        ----------
        action_sequence: action_sequence
            The action_sequence containing action sequence to be encoded

        Returns
        -------
        depth: torch.Tensor
            The depth of each action. The shape is (len(action_sequence),).
        adjacency_matrix: torch.Tensor
            The encoded tensor. The shape of tensor is
            (len(action_sequence), len(action_sequence)). If i th action is
            a parent of j th action, (i, j) element will be 1. the element
            will be 0 otherwise.
        """
        L = len(action_sequence.action_sequence)
        depth = torch.zeros(L)
        m = torch.zeros(L, L)

        for i in range(L):
            p = action_sequence.parent(i)
            if p is not None:
                depth[i] = depth[p.action] + 1
                m[p.action, i] = 1

        return depth, m

    def encode_each_action(self,
                           action_sequence: ActionSequence, query: List[str],
                           max_arity: int) \
            -> torch.Tensor:
        """
        Return the tensor encoding the each action

        Parameters
        ----------
        action_sequence: action_sequence
            The action_sequence containing action sequence to be encoded
        query: List[str]]
        max_arity: int

        Returns
        -------
        torch.Tensor
            The encoded tensor. The shape of tensor is
            (len(action_sequence), max_arity + 1, 3).
            [:, 0, 0] encodes the parent node type. [:, i, 0] encodes
            the node type of (i - 1)-th child node. [:, i, 1] encodes
            the token of (i - 1)-th child node. [:, i, 2] encodes the query
            index of (i - 1)-th child node.
            The padding value is -1.
        """
        L = len(action_sequence.action_sequence)
        retval = torch.ones(L, max_arity + 1, 3).long() * -1
        for i, action in enumerate(action_sequence.action_sequence):
            if isinstance(action, ApplyRule):
                if isinstance(action.rule, ExpandTreeRule):
                    # Encode parent
                    retval[i, 0, 0] = \
                        self._node_type_encoder.encode(action.rule.parent)
                    # Encode children
                    for j, (_, child) in enumerate(
                            action.rule.children[:max_arity]):
                        retval[i, j + 1, 0] = \
                            self._node_type_encoder.encode(child)
            else:
                gentoken: GenerateToken = action
                token = gentoken.token
                encoded_token = int(self._token_encoder.encode(token).numpy())

                if encoded_token != 0:
                    retval[i, 1, 1] = encoded_token

                if token.raw_value in query:
                    retval[i, 1, 2] = query.index(cast(str, token.raw_value))

        return retval

    def encode_path(self, action_sequence: ActionSequence, max_depth: int) \
            -> torch.Tensor:
        """
        Return the tensor encoding the each action

        Parameters
        ----------
        action_sequence: action_sequence
            The action_sequence containing action sequence to be encoded
        max_depth: int

        Returns
        -------
        torch.Tensor
            The encoded tensor. The shape of tensor is
            (len(action_sequence), max_depth).
            [i, :] encodes the path from the root node to i-th node.
            Each node represented by the rule id.
            The padding value is -1.
        """
        L = len(action_sequence.action_sequence)
        retval = torch.ones(L, max_depth).long() * -1
        for i in range(L):
            parent_opt = action_sequence.parent(i)
            if parent_opt is not None:
                p = action_sequence.action_sequence[parent_opt.action]
                if isinstance(p, ApplyRule):
                    retval[i, 0] = self._rule_encoder.encode(p.rule)
                retval[i, 1:] = retval[parent_opt.action, :max_depth - 1]

        return retval
