import torch
import numpy as np
from torchnlp.encoders import LabelEncoder
from typing import Callable, List, Any, Optional, Dict, TypeVar, Generic, cast
from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Token
from mlprogram.utils import Query

Input = TypeVar("Input")


class TransformQuery(Generic[Input]):
    def __init__(self, extract_query: Callable[[Input], Query],
                 word_encoder: LabelEncoder):
        self.extract_query = extract_query
        self.word_encoder = word_encoder

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        input = cast(Input, entry["input"])
        query = self.extract_query(input)

        entry["reference"] = query.reference
        entry["word_nl_query"] = \
            self.word_encoder.batch_encode(query.query_for_dnn)

        return entry


class TransformActionSequence:
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

        entry["actions"] = action_tensor
        entry["previous_actions"] = prev_action
        if self.train or "history" not in entry:
            entry["history"] = None
        if self.train or "hidden_state" not in entry:
            entry["hidden_state"] = None
        if self.train or "state" not in entry:
            entry["state"] = None

        return entry
