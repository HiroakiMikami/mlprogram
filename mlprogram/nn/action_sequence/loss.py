import torch
import torch.nn as nn

from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Loss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(Loss, self).__init__()
        self.reduction = reduction
        assert self.reduction == "mean" or self.reduction == "sum" or \
            self.reduction == "none"

    def forward(self,
                rule_probs: PaddedSequenceWithMask,
                token_probs: PaddedSequenceWithMask,
                reference_probs: PaddedSequenceWithMask,
                ground_truth_actions: PaddedSequenceWithMask
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        rule_probs: PaddedSequenceWithMask
            The probabilities of apply-rule. The shape is (L_a, B, num_rules).
        token_probs: PaddedSequenceWithMask
            The probabilities of gen-token. The shape is (L_a, B, num_tokens).
        reference_probs: PaddedSequenceWithMask
            The probabilities of reference-token. The shape is
            (L_a, B, reference_length).
        ground_truth_actions: PaddedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the reference).
            The padding value should be -1.
        """
        L_a, B, num_rules = rule_probs.data.shape
        _, _, num_tokens = token_probs.data.shape
        _, _, reference_length = reference_probs.data.shape

        gt_rule, gt_token, gt_reference = torch.split(
            ground_truth_actions.data, 1, dim=2)  # (L_a, B, 1)
        gt_rule = gt_rule.reshape([L_a, B])  # (L_a, B)
        gt_token = gt_token.reshape([L_a, B])  # (L_a, B)
        gt_reference = gt_reference.reshape([L_a, B])  # (L_a, B)
        # Change padding value
        rule = gt_rule + (gt_rule == -1).long() * (num_rules + 1)
        token = gt_token + (gt_token == -1).long() * (num_tokens + 1)
        reference = gt_reference + \
            (gt_reference == -1).long() * (reference_length) + 1

        device = gt_rule.device

        rule = torch.eye(num_rules + 1,
                         device=device)[gt_rule]  # (L_a, B, num_rules + 1)
        rule = rule[:, :, :-1]  # (L_a, B, num_rules)
        # (L_a, B, num_tokens + 1)
        token = torch.eye(num_tokens + 1,
                          device=device)[gt_token]
        token = token[:, :, :-1]  # (L_a, B, num_tokens)
        # (L_a, B, reference_length + 1)
        reference = torch.eye(reference_length + 1,
                              device=device)[gt_reference]
        # (L_a, B, reference_length)
        reference = reference[:, :, :-1]

        rule_prob_tensor = rule_probs.data * rule  # (L_a, B, num_rules)
        rule_prob_tensor = torch.sum(rule_prob_tensor, dim=2)  # (L_a, B)
        token_prob_tensor = token_probs.data * token  # (L_a, B, num_tokens)
        token_prob_tensor = torch.sum(token_prob_tensor, dim=2)  # (L_a, B)
        reference_prob_tensor = \
            reference_probs.data * reference  # (L_a, B, reference_length)
        reference_prob_tensor = \
            torch.sum(reference_prob_tensor, dim=2)  # (L_a, B)

        prob = rule_prob_tensor + token_prob_tensor + reference_prob_tensor
        prob = prob + (prob < 1e-7).to(rule_prob_tensor.dtype) * \
            1e-7  # avoid zero division

        likelihood = torch.log(prob)  # (L_a, B)
        loss = -likelihood * \
            ground_truth_actions.mask.to(rule_prob_tensor.dtype)  # (L_a, B)
        if self.reduction == "mean":
            return torch.mean(torch.sum(loss, dim=0))
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return torch.sum(loss, dim=0)


class EntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        assert self.reduction == "mean" or self.reduction == "sum" or \
            self.reduction == "none"

    def forward(self,
                rule_probs: PaddedSequenceWithMask,
                token_probs: PaddedSequenceWithMask,
                reference_probs: PaddedSequenceWithMask,
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        rule_probs: PaddedSequenceWithMask
            The probabilities of apply-rule. The shape is (L_a, B, num_rules).
        token_probs: PaddedSequenceWithMask
            The probabilities of gen-token. The shape is (L_a, B, num_tokens).
        reference_probs: PaddedSequenceWithMask
            The probabilities of reference-token. The shape is
            (L_a, B, reference_length).
        """
        def entropy(p: PaddedSequenceWithMask) -> torch.Tensor:
            log_p = torch.log(
                torch.where(p.mask[:, :, None], p.data, torch.zeros_like(p.data)) + 1e-7
            )
            out = -(p.data * log_p)
            out = out.sum(dim=2)
            torch.where(p.mask, out, torch.zeros_like(out))
            return out.sum(dim=0), p.mask.long().sum(dim=0)

        rule, n1 = entropy(rule_probs)
        token, n2 = entropy(token_probs)
        reference, n3 = entropy(reference_probs)

        out = (rule + token + reference) / (n1 + n2 + n3)

        if self.reduction == "mean":
            return torch.mean(out)
        elif self.reduction == "sum":
            return torch.sum(out)
        else:
            return out
