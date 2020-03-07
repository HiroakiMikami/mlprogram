import torch
from typing import List, Tuple

from nl2prog.nn.utils import rnn
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class Collate:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, data: List[Tuple[PaddedSequenceWithMask,
                                        Tuple[PaddedSequenceWithMask,
                                              PaddedSequenceWithMask],
                                        PaddedSequenceWithMask]]):
        inputs = []
        actions = []
        prev_actions = []
        ground_truths = []
        for input, action_sequence, ground_truth in data:
            action, prev_action = action_sequence
            inputs.append(input)
            actions.append(action)
            prev_actions.append(prev_action)
            ground_truths.append(ground_truth)
        inputs = rnn.pad_sequence(inputs, padding_value=-1)
        actions = rnn.pad_sequence(actions, padding_value=-1)
        prev_actions = rnn.pad_sequence(prev_actions, padding_value=-1)
        ground_truths = rnn.pad_sequence(ground_truths, padding_value=-1)

        return (inputs.to(self.device),
                (actions.to(self.device), prev_actions.to(self.device)),
                ground_truths.to(self.device))
