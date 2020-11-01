from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class RnnDecoder(nn.Module):
    def __init__(self, input_feature_size: int, action_feature_size: int,
                 output_feature_size: int, dropout: float = 0.0):
        super().__init__()
        self.output_feature_size = output_feature_size
        self.lstm = nn.LSTMCell(input_feature_size + action_feature_size,
                                output_feature_size)

    def forward(self, inputs: Environment) -> Environment:
        """
        Parameters
        ----------
        input_feature: torch.Tensor
        action_features: rnn.PackedSequenceWithMask
            The input sequence of feature vectors.
            The shape of input is (L_a, B, input_size)
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
        action_features = cast(PaddedSequenceWithMask,
                               inputs.states["action_features"])
        input_feature = cast(torch.Tensor, inputs.states["input_feature"])
        h_n = inputs.states["hidden_state"]
        c_n = inputs.states["state"]
        B = input_feature.data.shape[0]
        if h_n is None:
            h_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        if c_n is None:
            c_n = torch.zeros(B, self.output_feature_size,
                              device=action_features.data.device)
        h_n = cast(torch.Tensor, h_n)
        c_n = cast(torch.Tensor, c_n)
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

        inputs.states["action_features"] = rnn.PaddedSequenceWithMask(
            hs, action_features.mask)
        inputs.states["hidden_state"] = h1
        inputs.states["state"] = c1

        return inputs
