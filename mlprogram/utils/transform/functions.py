import torch
import numpy as np

from mlprogram.utils import Token
from mlprogram.utils.data import ListDataset
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.actions import ActionSequence
from typing import List, Callable, Any, Optional, Dict, cast, Generic, TypeVar


Code = TypeVar("Code")


class TransformCode(Generic[Code]):
    def __init__(self,
                 to_action_sequence: Callable[[Code],
                                              Optional[ActionSequence]]):
        self.to_action_sequence = to_action_sequence

    def __call__(self, **entry: Any) -> Optional[Dict[str, Any]]:
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

    def __call__(self, **entry: Any) -> Optional[Dict[str, Any]]:
        action_sequence = cast(ActionSequence, entry["action_sequence"])
        reference = cast(List[Token[str]], entry["reference"])
        # TODO use type in encoding action sequence
        a = self.action_sequence_encoder.encode_action(
            action_sequence, list(map(lambda x: x.value, reference)))
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        ground_truth = a[1:-1, 1:]
        entry["ground_truth_actions"] = ground_truth
        return entry


class RandomChoice:
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random
        self.rng = rng

    def __call__(self, **entry: Any) -> Dict[str, Any]:
        output = {}
        for key, value in entry.items():
            output[key] = self.rng.choice(value, size=()).item()
        return output


class TransformDataset:
    def __init__(self,
                 transform_input,
                 transform_code,
                 transform_action_sequence,
                 transform_ground_truth):
        self.choice = RandomChoice()
        self.transform_input = transform_input
        self.transform_code = transform_code
        self.transform_action_sequence = transform_action_sequence
        self.transform_ground_truth = transform_ground_truth

    def __call__(self, dataset: torch.utils.data.Dataset) \
            -> torch.utils.data.Dataset:
        entries = []
        for group in dataset:
            entry = self.choice(**group)
            entry = self.transform_input(**entry)
            entry = self.transform_code(**entry)
            if entry is None:
                continue
            entry = self.transform_action_sequence(**entry)
            if entry is None:
                continue
            entry = self.transform_ground_truth(**entry)
            if entry is None:
                continue
            entries.append(entry)
        return ListDataset(entries)
