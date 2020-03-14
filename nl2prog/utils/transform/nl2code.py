import torch
import numpy as np
from torchnlp.encoders import LabelEncoder
from typing import Callable, List, Any, Optional, Tuple, Union
from nl2prog.language.evaluator import Evaluator
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils import Query


class TransformQuery:
    def __init__(self, tokenize_query: Callable[[str], Query],
                 word_encoder: LabelEncoder):
        self.tokenize_query = tokenize_query
        self.word_encoder = word_encoder

    def __call__(self, query: Union[str, List[str]]) -> Tuple[List[str], Any]:
        if isinstance(query, str):
            query = self.tokenize_query(query)
        else:
            q = Query([], [])
            for word in query:
                q2 = self.tokenize_query(word)
                q.query_for_dnn.extend(q2.query_for_dnn)
                q.query_for_synth.extend(q2.query_for_synth)
            query = q

        return query.query_for_synth, \
            self.word_encoder.batch_encode(query.query_for_dnn)


class TransformEvaluator:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.train = train

    def __call__(self, evaluator: Evaluator, query_for_synth: List[str]) \
            -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], None]]:
        a = self.action_sequence_encoder.encode_action(evaluator,
                                                       query_for_synth)
        p = self.action_sequence_encoder.encode_parent(evaluator)
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

        return (action_tensor, prev_action), None
