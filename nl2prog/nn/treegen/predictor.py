import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn import PointerNet
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class Predictor(nn.Module):
    def __init__(self, feature_size: int, nl_feature_size: int,
                 rule_size: int, hidden_size: int):
        super(Predictor, self).__init__()
        self.select_rule = nn.Linear(feature_size, 1)
        self.rule = nn.Linear(feature_size, rule_size)
        self.copy = PointerNet(feature_size, nl_feature_size, hidden_size)

    def forward(self, feature: PaddedSequenceWithMask,
                nl_feature: PaddedSequenceWithMask) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        """
        Parameters
        ----------
        feature: PaddedSequenceWithMask
            (L_ast, N, feature_size) where L_ast is the sequence length,
            N is the batch size.
        nl_feature: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.

        Returns
        -------
        log_rule_prob: PaddedSequenceWithMask
            (L_ast, N, rule_size + L_nl) where L_ast is the sequence length,
            N is the batch_size.
        log_copy_prob: PaddedSequenceWithMask
            (L_ast, N, L_nl) where L_ast is the sequence length,
            N is the batch_size.
        """
        rule_pred = self.rule(feature.data)
        rule_log_prob = torch.log_softmax(rule_pred, dim=2)

        select_rule = self.select_rule(feature.data)
        select_rule_prob = torch.sigmoid(select_rule)

        copy_log_prob = self.copy(feature.data, nl_feature)

        rule_log_prob = torch.log(select_rule_prob) + rule_log_prob
        copy_log_prob = torch.log(1 - select_rule_prob) + copy_log_prob

        return PaddedSequenceWithMask(rule_log_prob, feature.mask), \
            PaddedSequenceWithMask(copy_log_prob, feature.mask)
