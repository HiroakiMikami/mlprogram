import torch
import torch.nn as nn
from typing import Dict, Any, cast

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

    def forward(self, **inputs: Any) \
            -> Dict[str, Any]:
        """
        Parameters
        ----------
        nl_query_features: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        action_features: PaddedSequenceWithMask
            (L_ast, N, feature_size) where L_ast is the sequence length,
            N is the batch size.

        Returns
        -------
        rule_probs: PaddedSequenceWithMask
            (L_ast, N, rule_size) where L_ast is the sequence length,
            N is the batch_size.
        token_probs: PaddedSequenceWithMask
           (L_ast, N, token_size) where L_ast is the sequence length,
            N is the batch_size.
        copy_probs: PaddedSequenceWithMask
            (L_ast, N, L_nl) where L_ast is the sequence length,
            N is the batch_size.
        """
        nl_query_features = cast(PaddedSequenceWithMask,
                                 inputs["nl_query_features"])
        action_features = cast(PaddedSequenceWithMask,
                               inputs["action_features"])
        rule_pred = self.rule(action_features.data)
        rule_prob = torch.softmax(rule_pred, dim=2)

        token_pred = self.token(action_features.data)
        token_prob = torch.softmax(token_pred, dim=2)

        select = self.select(action_features.data)
        select_prob = torch.softmax(select, dim=2)

        copy_log_prob = self.copy(action_features.data, nl_query_features)
        copy_prob = torch.exp(copy_log_prob)

        rule_log_prob = select_prob[:, :, 0:1] * rule_prob
        token_log_prob = select_prob[:, :, 1:2] * token_prob
        copy_log_prob = select_prob[:, :, 2:3] * copy_prob
        if self.training:
            inputs["rule_probs"] = \
                PaddedSequenceWithMask(rule_log_prob, action_features.mask)
            inputs["token_probs"] = \
                PaddedSequenceWithMask(token_log_prob, action_features.mask)
            inputs["copy_probs"] = \
                PaddedSequenceWithMask(copy_log_prob, action_features.mask)
        else:
            inputs["rule_probs"] = rule_log_prob[-1, :, :]
            inputs["token_probs"] = token_log_prob[-1, :, :]
            inputs["copy_probs"] = copy_log_prob[-1, :, :]

        return inputs
