import torch
from typing import Optional, cast, List, TypeVar, Generic
from mlprogram import Environment
from mlprogram.languages import Token
from mlprogram.languages import Parser
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.actions import ActionSequence
import numpy as np


Code = TypeVar("Code")


class AddEmptyReference(object):
    def __call__(self, entry: Environment) -> Optional[Environment]:
        entry.states["reference"] = []
        entry.states["reference_features"] = torch.zeros((0, 1))
        return entry


class GroundTruthToActionSequence(Generic[Code]):
    def __init__(self, parser: Parser[Code]):
        self.parser = parser

    def __call__(self, entry: Environment) -> Optional[Environment]:
        code = cast(Code, entry.supervisions["ground_truth"])
        ast = self.parser.parse(code)
        if ast is None:
            return None
        seq = ActionSequence.create(ast)
        entry.supervisions["action_sequence"] = seq
        return entry


class EncodeActionSequence:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder):

        self.action_sequence_encoder = action_sequence_encoder

    def __call__(self, entry: Environment) -> Optional[Environment]:
        action_sequence = cast(ActionSequence,
                               entry.supervisions["action_sequence"])
        reference = cast(List[Token[str, str]], entry.states["reference"])
        a = self.action_sequence_encoder.encode_action(
            action_sequence, reference)
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        ground_truth = a[1:-1, 1:]
        entry.supervisions["ground_truth_actions"] = ground_truth
        return entry


class AddPreviousActions:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True,
                 use_last: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train
        self.use_last = use_last  # TODO

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train:
            action_sequence = cast(ActionSequence,
                                   entry.supervisions["action_sequence"])
        else:
            action_sequence = cast(ActionSequence,
                                   entry.states["action_sequence"])
        reference = cast(List[Token[str, str]], entry.states["reference"])
        a = self.action_sequence_encoder.encode_action(
            action_sequence, reference)
        if a is None:
            return None
        if self.train:
            if np.any(a[-1, :].numpy() != -1):
                return None
            prev_action = a[:-2, 1:]
        else:
            prev_action = a[:-1, 1:]
            if self.use_last:
                prev_action = prev_action[-1, :].view(1, -1)

        entry.states["previous_actions"] = prev_action

        return entry


class AddActions:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True, use_last: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train
        self.use_last = use_last

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train:
            action_sequence = cast(ActionSequence,
                                   entry.supervisions["action_sequence"])
        else:
            action_sequence = cast(ActionSequence,
                                   entry.states["action_sequence"])
        reference = cast(List[Token[str, str]], entry.states["reference"])
        a = self.action_sequence_encoder.encode_action(
            action_sequence, reference)
        p = self.action_sequence_encoder.encode_parent(action_sequence)
        if a is None:
            return None
        if self.train:
            if np.any(a[-1, :].numpy() != -1):
                return None
            action_tensor = torch.cat(
                [a[1:-1, 0].view(-1, 1), p[1:-1, 1:3].view(-1, 2)],
                dim=1)
        else:
            action_tensor = torch.cat(
                [a[1:, 0].view(-1, 1), p[1:, 1:3].view(-1, 2)], dim=1)
            if self.use_last:
                action_tensor = action_tensor[-1, :].view(1, -1)

        entry.states["actions"] = action_tensor

        return entry


class AddPreviousActionRules:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_arity: int, train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.max_arity = max_arity
        self.train = train

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train:
            action_sequence = cast(ActionSequence,
                                   entry.supervisions["action_sequence"])
        else:
            action_sequence = cast(ActionSequence,
                                   entry.states["action_sequence"])
        reference = cast(List[Token[str, str]], entry.states["reference"])
        rule_prev_action = \
            self.action_sequence_encoder.encode_each_action(
                action_sequence, reference, self.max_arity)
        if self.train:
            rule_prev_action = rule_prev_action[:-1]

        entry.states["previous_action_rules"] = rule_prev_action
        return entry


class AddActionSequenceAsTree:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train:
            action_sequence = cast(ActionSequence,
                                   entry.supervisions["action_sequence"])
        else:
            action_sequence = cast(ActionSequence,
                                   entry.states["action_sequence"])
        depth, matrix = self.action_sequence_encoder.encode_tree(
            action_sequence)
        if self.train:
            depth = depth[:-1]
            matrix = matrix[:-1, :-1]

        entry.states["adjacency_matrix"] = matrix
        entry.states["depthes"] = depth

        return entry


class AddQueryForTreeGenDecoder:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_depth: int, train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.max_depth = max_depth
        self.train = train

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train:
            action_sequence = cast(ActionSequence,
                                   entry.supervisions["action_sequence"])
        else:
            action_sequence = cast(ActionSequence,
                                   entry.states["action_sequence"])
        query = \
            self.action_sequence_encoder.encode_path(
                action_sequence, self.max_depth)
        if self.train:
            query = query[:-1, :]

        entry.states["action_queries"] = query

        return entry


class AddStateForRnnDecoder:
    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train or "hidden_state" not in entry.states:
            entry.states["hidden_state"] = None
        if self.train or "state" not in entry.states:
            entry.states["state"] = None

        return entry


class AddHistoryState:
    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, entry: Environment) -> Optional[Environment]:
        if self.train or "history" not in entry.states:
            entry.states["history"] = None

        return entry
