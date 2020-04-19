import torch
import torch.nn.functional as F
from typing import List, Tuple

from nl2prog.nn.utils import rnn
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class CollateInput:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, inputs: List[Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        words = []
        chars = []
        for word, char in inputs:
            words.append(word)
            chars.append(char)
        pad_words = rnn.pad_sequence(words, padding_value=-1)
        pad_chars = rnn.pad_sequence(chars, padding_value=-1)

        return (pad_words.to(self.device), pad_chars.to(self.device))


class CollateActionSequence:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, data: List[Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]]) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask,
                     torch.Tensor, torch.Tensor]:
        prev_actions = []
        rule_prev_actions = []
        depths = []
        matrixs = []
        for prev_action, rule_prev_action, depth, matrix in data:
            prev_actions.append(prev_action)
            rule_prev_actions.append(rule_prev_action)
            depths.append(depth)
            matrixs.append(matrix)
        pad_prev_actions = rnn.pad_sequence(prev_actions, padding_value=-1)
        pad_rule_prev_actions = rnn.pad_sequence(rule_prev_actions,
                                                 padding_value=-1)
        pad_depths = rnn.pad_sequence(depths).data
        pad_depths = pad_depths.reshape(pad_depths.shape[1], -1).permute(1, 0)
        L = pad_prev_actions.data.shape[0]
        matrixs = [F.pad(m, (0, L - m.shape[0], 0, L - m.shape[1]))
                   for m in matrixs]
        pad_matrix = rnn.pad_sequence(matrixs).data.permute(1, 0, 2)
        return (pad_prev_actions.to(self.device),
                pad_rule_prev_actions.to(self.device),
                pad_depths.to(self.device), pad_matrix.to(self.device))


class CollateQuery:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, queries: List[torch.Tensor]) -> PaddedSequenceWithMask:
        pad_queries = rnn.pad_sequence(queries, padding_value=-1)

        return pad_queries.to(self.device)
