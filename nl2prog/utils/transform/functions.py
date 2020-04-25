import torch
import numpy as np

from nl2prog.utils.data import ListDataset
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.language.action import ActionSequence
from nl2prog.language.evaluator import Evaluator
from typing import List, Callable, Tuple, TypeVar, Generic, Optional

Code = TypeVar("Code")
Input = TypeVar("Input")
EncodedInput = TypeVar("EncodedInput")
EncodedActionSequence = TypeVar("EncodedActionSequence")
EncodedQuery = TypeVar("EncodedQuery")


class TransformCode(Generic[Code]):
    def __init__(self,
                 to_action_sequence: Callable[[Code],
                                              Optional[ActionSequence]]):
        self.to_action_sequence = to_action_sequence

    def __call__(self, code: Code) -> Optional[Evaluator]:
        action_sequence = self.to_action_sequence(code)
        if action_sequence is None:
            return None
        evaluator = Evaluator(options=action_sequence.options)
        for action in action_sequence.sequence:
            evaluator.eval(action)
        return evaluator


class TransformGroundTruth:
    def __init__(self,
                 action_sequence_encoder: ActionSequenceEncoder):

        self.action_sequence_encoder = action_sequence_encoder

    def __call__(self, evaluator: Evaluator, query_for_synth: List[str]) \
            -> Optional[torch.Tensor]:
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
                 transform_input: Callable[[Input],
                                           Tuple[List[str], EncodedInput]],
                 transform_code: Callable[[Code], Optional[Evaluator]],
                 transform_evaluator: Callable[
                     [Evaluator, List[str]],
                     Optional[Tuple[EncodedActionSequence, EncodedQuery]]],
                 transform_ground_truth: Callable[[Evaluator, List[str]],
                                                  Optional[torch.Tensor]]):
        self.transform_input = transform_input
        self.transform_code = transform_code
        self.transform_evaluator = transform_evaluator
        self.transform_ground_truth = transform_ground_truth

    def __call__(self, dataset: torch.utils.data.Dataset) \
            -> torch.utils.data.Dataset:
        entries = []
        for group in dataset:
            for entry in group:
                query_for_synth, input_tensor = \
                    self.transform_input(entry.input)
                evaluator = self.transform_code(entry.ground_truth)
                if evaluator is None:
                    continue
                tmp = self.transform_evaluator(
                    evaluator, query_for_synth)
                ground_truth = self.transform_ground_truth(
                    evaluator, query_for_synth)
                if ground_truth is None or tmp is None:
                    continue
                action_sequence, query = tmp
                entries.append((input_tensor, action_sequence, query,
                                ground_truth))
        return ListDataset(entries)
