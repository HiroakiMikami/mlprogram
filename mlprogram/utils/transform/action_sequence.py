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


class TransformCode(Generic[Code]):
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


class TransformGroundTruth:
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


class TransformActionSequenceForRnnDecoder:
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
            prev_action = a[-2, 1:].view(1, -1)

        entry.states["previous_actions"] = prev_action
        if self.train or "hidden_state" not in entry.states:
            entry.states["hidden_state"] = None
        if self.train or "state" not in entry.states:
            entry.states["state"] = None

        return entry
