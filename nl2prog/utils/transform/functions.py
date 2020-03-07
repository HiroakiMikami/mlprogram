import torch
from torchnlp.encoders import LabelEncoder
import numpy as np

from nl2prog.utils import Query
from nl2prog.utils.data import ListDataset
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.language.action import ActionOptions, ActionSequence
from nl2prog.language.evaluator import Evaluator
from typing import List, Union, Callable, Tuple, Any, Optional


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


class TransformGroundTruth:
    def __init__(self,
                 to_action_sequence: Callable[[Any],
                                              Union[ActionSequence, None]],
                 action_sequence_encoder: ActionSequenceEncoder,
                 options: ActionOptions = ActionOptions(True, True)):
        self.to_action_sequence = to_action_sequence
        self.action_sequence_encoder = action_sequence_encoder
        self.options = options

    def __call__(self, code: Any, query_for_synth: List[str]) \
            -> Optional[torch.Tensor]:
        action_sequence = self.to_action_sequence(code)
        if action_sequence is None:
            return None
        evaluator = Evaluator(options=self.options)
        for action in action_sequence:
            evaluator.eval(action)
        a = self.action_sequence_encoder.encode_action(evaluator,
                                                       query_for_synth)
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        ground_truth = a[1:-1, 1:]
        return ground_truth


class TransformDataset:
    def __init__(self,
                 transform_input: Callable[[Any], Tuple[List[str], Any]],
                 transform_code: Callable[[Any, List[str]], Optional[Any]],
                 transform_ground_truth: Callable[[Any, List[str]],
                                                  Optional[torch.Tensor]]):
        self.transform_input = transform_input
        self.transform_code = transform_code
        self.transform_ground_truth = transform_ground_truth

    def __call__(self, dataset: torch.utils.data.Dataset) \
            -> torch.utils.data.Dataset:
        entries = []
        for group in dataset:
            for entry in group:
                query_for_synth, input_tensor = \
                    self.transform_input(entry.query)
                action_sequence = self.transform_code(
                    entry.ground_truth, query_for_synth)
                ground_truth = self.transform_ground_truth(
                    entry.ground_truth, query_for_synth)
                if action_sequence is None or ground_truth is None:
                    continue
                entries.append((input_tensor, action_sequence, ground_truth))
        return ListDataset(entries)
