import torch
import torch.nn as nn

from nl2code.nn.utils.rnn import PaddedSequenceWithMask


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self,
                rule_prob: PaddedSequenceWithMask,
                token_prob: PaddedSequenceWithMask,
                copy_prob: PaddedSequenceWithMask,
                ground_truth_action: PaddedSequenceWithMask):
        """
        Parameters
        ----------
        rule_prob: PaddedSequenceWithMask
            The probabilities of apply-rule. The shape is (L_a, B, num_rules).
        token_pred: PaddedSequenceWithMask
            The probabilities of gen-token. The shape is (L_a, B, num_tokens).
        copy_pred: PaddedSequenceWithMask
            The probabilities of copy-token. The shape is
            (L_a, B, query_length).
        ground_truth_action: PaddedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).
            The padding value should be -1.
        """
        L_a, B, num_rules = rule_prob.data.shape
        _, _, num_tokens = token_prob.data.shape
        _, _, query_length = copy_prob.data.shape

        gt_rule, gt_token, gt_copy = torch.split(
            ground_truth_action.data, 1, dim=2)  # (L_a, B, 1)
        gt_rule = gt_rule.reshape([L_a, B])  # (L_a, B)
        gt_token = gt_token.reshape([L_a, B])  # (L_a, B)
        gt_copy = gt_copy.reshape([L_a, B])  # (L_a, B)

        _, rule_pred = torch.max(rule_prob.data, 2)  # (L_a, B)
        _, token_pred = torch.max(token_prob.data, 2)  # (L_a, B)
        _, copy_pred = torch.max(copy_prob.data, 2)  # (L_a, B)

        n_rule = (gt_rule != -1).long().sum()
        n_token = (gt_token != -1).long().sum()
        n_copy = (gt_copy != -1).long().sum()
        rule_acc = ((rule_pred == gt_rule) * (gt_rule != -1).long()).sum()
        token_acc = ((token_pred == gt_token) * (gt_token != -1).long()).sum()
        copy_acc = ((copy_pred == gt_copy) * (gt_copy != -1).long()).sum()

        return (rule_acc + token_acc + copy_acc).float() \
            / (n_rule + n_token + n_copy)
