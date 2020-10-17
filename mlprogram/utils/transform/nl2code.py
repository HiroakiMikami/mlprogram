import torch
import numpy as np
from torchnlp.encoders import LabelEncoder
from typing import Callable, List, Optional, TypeVar, Generic, cast
from mlprogram import Environment
from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Token

Input = TypeVar("Input")


class TransformQuery(Generic[Input]):
    def __init__(self, extract_reference: Callable[[Input], List[Token]],
                 word_encoder: LabelEncoder):
        self.extract_reference = extract_reference
        self.word_encoder = word_encoder

    def __call__(self, entry: Environment) -> Environment:
        input = cast(Input, entry.inputs["input"])
        reference = self.extract_reference(input)

        entry.states["reference"] = reference
        entry.states["word_nl_query"] = self.word_encoder.batch_encode([
            token.value for token in reference
        ])

        return entry


class TransformActionSequence:
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
        p = self.action_sequence_encoder.encode_parent(action_sequence)
        if a is None:
            return None
        if self.train:
            if np.any(a[-1, :].numpy() != -1):
                return None
            action_tensor = torch.cat(
                [a[1:-1, 0].view(-1, 1), p[1:-1, 1:3].view(-1, 2)],
                dim=1)
            prev_action = a[:-2, 1:]
        else:
            action_tensor = torch.cat(
                [a[-1, 0].view(1, -1), p[-1, 1:3].view(1, -1)], dim=1)
            prev_action = a[-2, 1:].view(1, -1)

        entry.states["actions"] = action_tensor
        entry.states["previous_actions"] = prev_action
        if self.train or "history" not in entry.states:
            entry.states["history"] = None
        if self.train or "hidden_state" not in entry.states:
            entry.states["hidden_state"] = None
        if self.train or "state" not in entry.states:
            entry.states["state"] = None

        return entry
