from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class InjectInput(nn.Module):
    def forward(self, input_size: int,
                action_size: int,
                hidden_state_size: int,
                state_size: int) -> Tuple[int, nn.Module]:
        raise NotImplementedError


class _CatInput(nn.Module):
    def forward(self, input_feature: torch.Tensor,
                action_feature: torch.Tensor,
                hidden_state: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        input_feature: torch.Tensor
            The shape is (batch_size, input_feature_size)
        aciton_feature: torch.Tensor
            The feature tensor of one Action
            The shape is (batch_size, action_feature_size)
        """
        return torch.cat([input_feature, action_feature], dim=1)


class CatInput(InjectInput):
    def forward(self, input_size: int,
                action_size: int,
                hidden_state_size: int,
                state_size: int) -> Tuple[int, nn.Module]:
        return (input_size + action_size, _CatInput())


class _AttentionInput(nn.Module):
    def __init__(self, input_size: int, attn_hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_size, attn_hidden_size),
            nn.Tanh(),
            nn.Linear(attn_hidden_size, 1)
        )

    def forward(self, input_feature: PaddedSequenceWithMask,
                action_feature: torch.Tensor,
                hidden_state: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        input_feature: PaddedSequenceWithMask
            The shape is (input_length, batch_size, input_feature_size)
        aciton_feature: torch.Tensor
            The feature tensor of one Action
            The shape is (batch_size, action_feature_size)
        """
        L, B, E = input_feature.data.shape
        query = hidden_state.reshape(1, *hidden_state.shape)
        query = torch.cat([query.expand(L, *query.shape[1:]), input_feature.data],
                          dim=2)
        logit = self.attn(query).squeeze(-1)  # (L, batch_size)
        # fill -inf to non-data element
        logit = torch.where(input_feature.mask != 0,
                            logit, torch.full_like(logit, -1e10))
        p = torch.softmax(logit, dim=0)  # (L, B)
        ctx = torch.sum(input_feature.data * p.reshape(L, B, 1), dim=0)

        return torch.cat([ctx, action_feature], dim=1)


class AttentionInput(InjectInput):
    def __init__(self, attn_hidden_size: int):
        super().__init__()
        self.attn_hidden_size = attn_hidden_size

    def forward(self, input_size: int,
                action_size: int,
                hidden_state_size: int,
                state_size: int) -> Tuple[int, nn.Module]:
        return (input_size + action_size,
                _AttentionInput(input_size + hidden_state_size, self.attn_hidden_size))


class LSTMDecoder(nn.Module):
    def __init__(self,
                 inject_input: InjectInput,
                 input_feature_size: int, action_feature_size: int,
                 output_feature_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.output_feature_size = output_feature_size
        output_size, self.inject_input = inject_input(
            input_feature_size,
            action_feature_size,
            output_feature_size,
            output_feature_size)
        self.lstm = nn.LSTMCell(output_size, output_feature_size)

    def forward(self,
                input_feature: Any,
                action_features: PaddedSequenceWithMask,
                hidden_state: Optional[torch.Tensor],
                state: Optional[torch.Tensor]
                ) -> Tuple[PaddedSequenceWithMask, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_feature: Any
        aciton_features: rnn.PaddedSequenceWithMask
            The feature tensor of ActionSequence
            The shape is (len(action_sequence) + 1, batch_size, action_feature_size)
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

        L_a, B, _ = action_features.data.shape

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
            x = nn.functional.dropout(d.squeeze(0), p=self.dropout)
            h = nn.functional.dropout(s[0], p=self.dropout)
            input = self.inject_input(input_feature, x, s[0], s[1])
            h1, c1 = self.lstm(input, (h, s[1]))
            hs.append(h1)
            cs.append(c1)
            s = (h1, c1)
        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return rnn.PaddedSequenceWithMask(hs, action_features.mask), h1, c1
