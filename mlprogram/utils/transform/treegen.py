import torch
import numpy as np
from typing import Callable, List, Any, Optional, Tuple, Dict
from torchnlp.encoders import LabelEncoder

from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.utils import Query


class TransformQuery:
    def __init__(self, extract_query: Callable[[Any], Query],
                 word_encoder: LabelEncoder, char_encoder: LabelEncoder,
                 max_word_length: int):
        self.extract_query = extract_query
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.max_word_length = max_word_length

    def __call__(self, input: Any) -> Tuple[List[str], Dict[str, Any]]:
        query = self.extract_query(input)

        word_query = self.word_encoder.batch_encode(query.query_for_dnn)
        char_query = \
            torch.ones(len(query.query_for_dnn), self.max_word_length).long() \
            * -1
        for i, word in enumerate(query.query_for_dnn):
            chars = self.char_encoder.batch_encode(word)
            length = min(self.max_word_length, len(chars))
            char_query[i, :length] = \
                self.char_encoder.batch_encode(word)[:length]
        return query.query_for_synth, {
            "word_nl_query": word_query,
            "char_nl_query": char_query
        }


class TransformEvaluator:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_arity: int, max_depth: int, train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.max_arity = max_arity
        self.max_depth = max_depth
        self.train = train

    def __call__(self, evaluator: ActionSequence, query_for_synth: List[str]) \
            -> Optional[Dict[str, Any]]:
        a = self.action_sequence_encoder.encode_action(evaluator,
                                                       query_for_synth)
        rule_prev_action = \
            self.action_sequence_encoder.encode_each_action(
                evaluator, query_for_synth, self.max_arity)
        path = \
            self.action_sequence_encoder.encode_path(evaluator, self.max_depth)
        depth, matrix = self.action_sequence_encoder.encode_tree(evaluator)
        if a is None:
            return None
        if self.train:
            if np.any(a[-1, :].numpy() != -1):
                return None
            prev_action = a[:-2, 1:]
            query = path[:-1, :]
            rule_prev_action = rule_prev_action[:-1]
            depth = depth[:-1]
            matrix = matrix[:-1, :-1]
        else:
            prev_action = a[:-1, 1:]
            query = path
            rule_prev_action = \
                self.action_sequence_encoder.encode_each_action(
                    evaluator, query_for_synth, self.max_arity)

        return {
            "previous_actions": prev_action,
            "previous_action_rules": rule_prev_action,
            "depthes": depth,
            "adjacency_matrix": matrix,
            "action_queries": query
        }
