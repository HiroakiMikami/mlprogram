import torch
import torch.nn as nn
from typing import Tuple

from mlprogram.nn import PointerNet
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Predictor(nn.Module):
    def __init__(self, feature_size: int, nl_feature_size: int,
                 rule_size: int, token_size: int, hidden_size: int):
        super(Predictor, self).__init__()
        self.select = nn.Linear(feature_size, 3)
        self.rule = nn.Linear(feature_size, rule_size)
        self.token = nn.Linear(feature_size, token_size)
        self.copy = PointerNet(feature_size, nl_feature_size, hidden_size)

    def forward(self, nl_feature: PaddedSequenceWithMask,
                feature: PaddedSequenceWithMask) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask,
                     PaddedSequenceWithMask]:
        """
        Parameters
        ----------
        nl_feature: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        feature: PaddedSequenceWithMask
            (L_ast, N, feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        rule_prob: PaddedSequenceWithMask
            (L_ast, N, rule_size) where L_ast is the sequence length,
            N is the batch_size.
        token_prob: PaddedSequenceWithMask
           (L_ast, N, token_size) where L_ast is the sequence length,
            N is the batch_size.
        copy_prob: PaddedSequenceWithMask
            (L_ast, N, L_nl) where L_ast is the sequence length,
            N is the batch_size.
        """
        rule_pred = self.rule(feature.data)
        rule_prob = torch.softmax(rule_pred, dim=2)

        token_pred = self.token(feature.data)
        token_prob = torch.softmax(token_pred, dim=2)

        select = self.select(feature.data)
        select_prob = torch.softmax(select, dim=2)

        copy_log_prob = self.copy(feature.data, nl_feature)
        copy_prob = torch.exp(copy_log_prob)

        rule_log_prob = select_prob[:, :, 0:1] * rule_prob
        token_log_prob = select_prob[:, :, 1:2] * token_prob
        copy_log_prob = select_prob[:, :, 2:3] * copy_prob

        return PaddedSequenceWithMask(rule_log_prob, feature.mask), \
            PaddedSequenceWithMask(token_log_prob, feature.mask), \
            PaddedSequenceWithMask(copy_log_prob, feature.mask)
