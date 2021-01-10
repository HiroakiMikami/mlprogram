from typing import Any, Generic, List, Optional, Tuple, TypeVar, cast

import numpy as np
import torch
from torch import nn

from mlprogram.actions import ActionSequence
from mlprogram.builtins import Environment
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Parser, Token

Code = TypeVar("Code")
Kind = TypeVar("Kind")
Value = TypeVar("Value")


class AddEmptyReference(nn.Module, Generic[Kind, Value]):
    def __call__(self) -> Tuple[List[Token[Kind, Value]], torch.Tensor]:
        return [], torch.zeros((0, 1))


class GroundTruthToActionSequence(nn.Module, Generic[Code]):
    def __init__(self, parser: Parser[Code]):
        super().__init__()
        self.parser = parser

    def forward(self, ground_truth: Code) -> ActionSequence:
        code = ground_truth
        ast = self.parser.parse(code)
        if ast is None:
            raise RuntimeError(f"cannot convert to ActionSequence: {ground_truth}")
        return cast(ActionSequence, ActionSequence.create(ast))


class EncodeActionSequence(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]]
                ) -> torch.Tensor:
        a = self.action_sequence_encoder.encode_action(action_sequence, reference)
        if a is None:
            raise RuntimeError("cannot encode ActionSequence")
        if np.any(a[-1, :].numpy() != -1):
            raise RuntimeError("ActionSequence is incomplete")
        ground_truth = a[1:-1, 1:]
        return ground_truth


class AddPreviousActions(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 n_dependent: Optional[int] = None):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder
        self.n_dependent = n_dependent

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]],
                train: bool) -> torch.Tensor:
        # TODO use self.training instead of train argument
        a = self.action_sequence_encoder.encode_action(action_sequence, reference)
        if a is None:
            raise RuntimeError("cannot encode ActionSequence")
        if train:
            if np.any(a[-1, :].numpy() != -1):
                raise RuntimeError("ActionSequence is incomplete")
            prev_action = a[:-2, 1:]
        else:
            prev_action = a[:-1, 1:]
            if self.n_dependent is not None:
                prev_action = prev_action[-self.n_dependent:, :]

        return prev_action


class AddActions(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 n_dependent: Optional[int] = None):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder
        self.n_dependent = n_dependent

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]],
                train: bool) -> torch.Tensor:
        a = self.action_sequence_encoder.encode_action(
            action_sequence, reference)
        p = self.action_sequence_encoder.encode_parent(action_sequence)
        if a is None:
            raise RuntimeError("cannot encode ActionSequence")
        if train:
            if np.any(a[-1, :].numpy() != -1):
                raise RuntimeError("cannot encode ActionSequence")
            action_tensor = torch.cat(
                [a[1:-1, 0].view(-1, 1), p[1:-1, 1:3].view(-1, 2)],
                dim=1)
        else:
            action_tensor = torch.cat(
                [a[1:, 0].view(-1, 1), p[1:, 1:3].view(-1, 2)], dim=1)
            if self.n_dependent is not None:
                action_tensor = action_tensor[-self.n_dependent:, :]

        return action_tensor


class AddPreviousActionRules(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_arity: int,
                 n_dependent: Optional[int] = None):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder
        self.max_arity = max_arity
        self.n_dependent = n_dependent

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]],
                train: bool) -> torch.Tensor:
        rule_prev_action = \
            self.action_sequence_encoder.encode_each_action(
                action_sequence, reference, self.max_arity)
        if train:
            rule_prev_action = rule_prev_action[:-1]
        else:
            if self.n_dependent is not None:
                rule_prev_action = rule_prev_action[-self.n_dependent:, :]

        return rule_prev_action


class AddActionSequenceAsTree(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]],
                train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        depth, matrix = self.action_sequence_encoder.encode_tree(
            action_sequence)
        if train:
            depth = depth[:-1]
            matrix = matrix[:-1, :-1]

        return matrix, depth


class AddQueryForTreeGenDecoder(nn.Module, Generic[Kind, Value]):
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_depth: int,
                 n_dependent: Optional[int] = None):
        super().__init__()
        self.action_sequence_encoder = action_sequence_encoder
        self.max_depth = max_depth
        self.n_dependent = n_dependent

    def forward(self,
                action_sequence: ActionSequence,
                reference: List[Token[Kind, Value]],
                train: bool) -> torch.Tensor:
        query = \
            self.action_sequence_encoder.encode_path(
                action_sequence, self.max_depth)
        if train:
            query = query[:-1, :]
        else:
            if self.n_dependent:
                query = query[-self.n_dependent:, :]

        return query


class AddState(nn.Module):
    def __init__(self, key: str, initial: Any = None):
        super().__init__()
        self.key = key
        self.initial = initial

    def __call__(self, entry: Environment) -> Environment:
        entry = cast(Environment, entry.clone())
        train = True
        if "train" in entry:
            train = entry["train"]
        if "action_sequence" in entry:
            train = entry.is_supervision("action_sequence")
        if train or self.key not in entry:
            entry[self.key] = self.initial

        return entry
