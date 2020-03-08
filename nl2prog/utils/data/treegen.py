import torch
import torch.nn.functional as F
from typing import List, Tuple

from nl2prog.nn.utils import rnn
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class Collate:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, data: List[Tuple[Tuple[PaddedSequenceWithMask,
                                              PaddedSequenceWithMask],
                                        Tuple[PaddedSequenceWithMask,
                                              PaddedSequenceWithMask,
                                              torch.Tensor, torch.Tensor],
                                        PaddedSequenceWithMask]]):
        words = []
        chars = []
        prev_actions = []
        rule_prev_actions = []
        ground_truths = []
        depths = []
        matrixs = []
        queries = []
        for input, action_sequence, query, ground_truth in data:
            word, char = input
            prev_action, rule_prev_action, depth, matrix = action_sequence
            words.append(word)
            chars.append(char)
            prev_actions.append(prev_action)
            rule_prev_actions.append(rule_prev_action)
            depths.append(depth)
            matrixs.append(matrix)
            queries.append(query)
            ground_truths.append(ground_truth)
        words = rnn.pad_sequence(words, padding_value=-1)
        chars = rnn.pad_sequence(chars, padding_value=-1)
        prev_actions = rnn.pad_sequence(prev_actions, padding_value=-1)
        rule_prev_actions = rnn.pad_sequence(rule_prev_actions,
                                             padding_value=-1)
        depths = rnn.pad_sequence(depths).data.reshape(1, -1).permute(1, 0)
        L = prev_actions.data.shape[0]
        matrixs = [F.pad(m, (0, L - m.shape[0], 0, L - m.shape[1]))
                   for m in matrixs]
        matrix = rnn.pad_sequence(matrixs).data.permute(1, 0, 2)
        queries = rnn.pad_sequence(queries, padding_value=-1)
        ground_truths = rnn.pad_sequence(ground_truths, padding_value=-1)

        return ((words.to(self.device), chars.to(self.device)),
                (prev_actions.to(self.device),
                 rule_prev_actions.to(self.device),
                 depths.to(self.device), matrix.to(self.device)),
                queries.to(self.device),
                ground_truths.to(self.device))
