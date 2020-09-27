import torch
from typing import Dict, Any, Optional, cast, List, TypeVar, Generic, Callable
from mlprogram.languages import Token
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.actions import ActionSequence
import numpy as np


Code = TypeVar("Code")


class AddEmptyReference(object):
    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entry["reference"] = []
        entry["reference_features"] = torch.zeros((0, 1))
        return entry


class TransformCode(Generic[Code]):
    def __init__(self,
                 to_action_sequence: Callable[[Code],
                                              Optional[ActionSequence]]):
        self.to_action_sequence = to_action_sequence

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        code = cast(Code, entry["ground_truth"])
        seq = self.to_action_sequence(code)
        if seq is None:
            return None
        entry["action_sequence"] = seq
        return entry


class TransformGroundTruth:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder):

        self.action_sequence_encoder = action_sequence_encoder

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action_sequence = cast(ActionSequence, entry["action_sequence"])
        reference = cast(List[Token[str, str]], entry["reference"])
        a = self.action_sequence_encoder.encode_action(
            action_sequence, reference)
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        ground_truth = a[1:-1, 1:]
        entry["ground_truth_actions"] = ground_truth
        return entry


class TransformActionSequenceForRnnDecoder:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train

    def __call__(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action_sequence = cast(ActionSequence, entry["action_sequence"])
        reference = cast(List[Token[str, str]], entry["reference"])
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

        entry["previous_actions"] = prev_action
        if self.train or "hidden_state" not in entry:
            entry["hidden_state"] = None
        if self.train or "state" not in entry:
            entry["state"] = None

        return entry
