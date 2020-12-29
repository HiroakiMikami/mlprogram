from typing import Optional, Tuple

import torch
import torch.nn as nn

from mlprogram.nn import EmbeddingWithMask
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class LSTMDecoder(nn.Module):
    def __init__(self,
                 n_rule: int,
                 n_token: int,
                 input_feature_size: int, action_feature_size: int,
                 output_feature_size: int, dropout: float = 0.0):
        super().__init__()
        self._rule_embed = EmbeddingWithMask(n_rule, action_feature_size, -1)
        self._token_embed = EmbeddingWithMask(n_token, action_feature_size, -1)
        self.output_feature_size = output_feature_size
        self.lstm = nn.LSTMCell(input_feature_size + action_feature_size,
                                output_feature_size)

    def forward(self,
                input_feature: torch.Tensor,
                previous_actions: PaddedSequenceWithMask,
                hidden_state: Optional[torch.Tensor],
                state: Optional[torch.Tensor]
                ) -> Tuple[PaddedSequenceWithMask, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_feature: torch.Tensor
        previous_acitons: rnn.PaddedSequenceWithMask
            The previous action sequence.
            The encoded tensor with the shape of
            (len(action_sequence) + 1, 3). Each action will be encoded by
            the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the reference).
            The padding value should be -1.
        hidden_state: torch.Tensor
            The LSTM initial hidden state. The shape is (B, hidden_size)
        state: torch.Tensor
            The LSTM initial state. The shape is (B, hidden_size)

        Returns
        -------
        action_features: PaddedSequenceWithMask
            Packed sequence containing the output hidden states.
        hidden_state: torch.Tensor
            The tuple of the next hidden state. The shape is (B, hidden_size)
        state: torch.Tensor
            The tuple of the next state. The shape is (B, hidden_size)
        """
        h_n = hidden_state
        c_n = state

        L_a, B, _ = previous_actions.data.shape
        prev_rules, prev_tokens, _ = torch.split(
            previous_actions.data, 1, dim=2)  # (L_a, B, 1)

        # Change the padding value
        prev_rules = prev_rules.reshape([L_a, B])
        prev_tokens = prev_tokens.reshape([L_a, B])

        # Embed previous actions
        action_features = self._rule_embed(prev_rules) + self._token_embed(prev_tokens)
        action_features = PaddedSequenceWithMask(action_features, previous_actions.mask)

        if h_n is None:
            h_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        if c_n is None:
            c_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        s = (h_n, c_n)
        hs = []
        cs = []
        for d in torch.split(action_features.data, 1, dim=0):
            input = torch.cat([input_feature, d.squeeze(0)], dim=1)
            h1, c1 = self.lstm(input, s)
            hs.append(h1)
            cs.append(c1)
            s = (h1, c1)
        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return rnn.PaddedSequenceWithMask(hs, action_features.mask), h1, c1
