import torch
import numpy as np
from torchnlp.encoders import LabelEncoder
from typing import Callable, List, Any, Optional, Tuple, Dict
from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Query


class TransformQuery:
    def __init__(self, extract_query: Callable[[Any], Query],
                 word_encoder: LabelEncoder):
        self.extract_query = extract_query
        self.word_encoder = word_encoder

    def __call__(self, input: Any) -> Tuple[List[str], Dict[str, Any]]:
        query = self.extract_query(input)

        return query.query_for_synth, \
            {"word_nl_query":
             self.word_encoder.batch_encode(query.query_for_dnn)}


class TransformActionSequence:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train

    def __call__(self, action_sequence: ActionSequence,
                 query_for_synth: List[str]) \
            -> Optional[Dict[str, Any]]:
        a = self.action_sequence_encoder.encode_action(action_sequence,
                                                       query_for_synth)
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

        return {
            "actions": action_tensor,
            "previous_actions": prev_action,
            "history": None,
            "hidden_state": None,
            "state": None
        }
