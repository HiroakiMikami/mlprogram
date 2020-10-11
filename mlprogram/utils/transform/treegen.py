import torch
import numpy as np
from typing import Callable, List, Optional, TypeVar, Generic, cast
from torchnlp.encoders import LabelEncoder

from mlprogram import Environment
from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Token


Input = TypeVar("Input")


class TransformQuery(Generic[Input]):
    def __init__(self, extract_reference: Callable[[Input], List[Token]],
                 word_encoder: LabelEncoder, char_encoder: LabelEncoder,
                 max_word_length: int):
        self.extract_reference = extract_reference
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.max_word_length = max_word_length

    def __call__(self, entry: Environment) -> Environment:
        input = cast(Input, entry.inputs["input"])
        reference = self.extract_reference(input)

        word_query = self.word_encoder.batch_encode([
            token.value for token in reference
        ])
        char_query = \
            torch.ones(len(reference), self.max_word_length).long() \
            * -1
        for i, token in enumerate(reference):
            chars = self.char_encoder.batch_encode(token.value)
            length = min(self.max_word_length, len(chars))
            char_query[i, :length] = chars[:length]
        entry.states["reference"] = reference
        entry.states["word_nl_query"] = word_query
        entry.states["char_nl_query"] = char_query

        return entry


class TransformActionSequence:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_arity: int, max_depth: int, train: bool = True):
        self.action_sequence_encoder = action_sequence_encoder
        self.max_arity = max_arity
        self.max_depth = max_depth
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
        rule_prev_action = \
            self.action_sequence_encoder.encode_each_action(
                action_sequence, reference, self.max_arity)
        path = \
            self.action_sequence_encoder.encode_path(
                action_sequence, self.max_depth)
        depth, matrix = self.action_sequence_encoder.encode_tree(
            action_sequence)
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
                    action_sequence, reference, self.max_arity)

        entry.states["previous_actions"] = prev_action
        entry.states["previous_action_rules"] = rule_prev_action
        entry.states["depthes"] = depth
        entry.states["adjacency_matrix"] = matrix
        entry.states["action_queries"] = query

        return entry
