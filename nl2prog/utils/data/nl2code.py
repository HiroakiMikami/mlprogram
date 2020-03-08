import torch
from typing import List, Tuple

from nl2prog.nn.utils import rnn
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class CollateInput:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, inputs: List[torch.Tensor]) -> PaddedSequenceWithMask:
        inputs = rnn.pad_sequence(inputs, padding_value=-1)

        return inputs.to(self.device)


class CollateActionSequence:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, action_sequence: List[Tuple[torch.Tensor,
                                                   torch.Tensor]]) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        actions = []
        prev_actions = []
        for action, prev_action in action_sequence:
            actions.append(action)
            prev_actions.append(prev_action)
        actions = rnn.pad_sequence(actions, padding_value=-1)
        prev_actions = rnn.pad_sequence(prev_actions, padding_value=-1)

        return (actions.to(self.device), prev_actions.to(self.device))


class CollateQuery:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, queries: List[None]) -> None:
        return None
