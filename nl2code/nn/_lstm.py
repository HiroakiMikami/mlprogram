import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LSTMCell(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool = True, dropout: float = 0.0):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias)
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x
        nn.init.orthogonal_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if bias:
            nn.init.zeros_(self.bias_hh)
            nn.init.zeros_(self.bias_ih)
            self.bias_ih.data[hidden_size:(2 * hidden_size)] = 1

    def forward(self, input: torch.FloatTensor,
                states: Tuple[torch.FloatTensor, torch.FloatTensor] = None) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if not self.training:
            return super().forward(input, states)

        weight_ih = self.weight_ih.view(4, self.hidden_size, -1)
        W_ii, W_if, W_ig, W_io = torch.split(weight_ih, 1, dim=0)
        weight_hh = self.weight_hh.view(4, self.hidden_size, -1)
        W_hi, W_hf, W_hg, W_ho = torch.split(weight_hh, 1, dim=0)
        if self.bias:
            bias_ih = self.bias_ih.view(4, -1)
            b_ii, b_if, b_ig, b_io = torch.split(bias_ih, 1, dim=0)
            bias_hh = self.bias_hh.view(4, -1)
            b_hi, b_hf, b_hg, b_ho = torch.split(bias_hh, 1, dim=0)
        else:
            b_ii = None
            b_if = None
            b_ig = None
            b_io = None
            b_hi = None
            b_hf = None
            b_hg = None
            b_ho = None

        dropout = self.dropout
        hidden_size = self.hidden_size

        B, _ = input.shape
        device = input.device
        if states is None:
            h_0 = torch.zeros(B, hidden_size, device=device)
            c_0 = torch.zeros(B, hidden_size, device=device)
        else:
            h_0, c_0 = states

        # (B, input_size + hidden_size)
        x_i = torch.cat([dropout(input), dropout(h_0)], dim=1)
        # (hidden_size, input_size + hidden_size)
        W_i = torch.cat([W_ii, W_hi], dim=2).view(hidden_size, -1)
        b_i = b_ii + b_hi if self.bias else None
        x_i = F.linear(x_i, W_i, b_i)
        i = torch.sigmoid(x_i)

        # (B, input_size + hidden_size)
        x_f = torch.cat([dropout(input), dropout(h_0)], dim=1)
        # (hidden_size, input_size + hidden_size)
        W_f = torch.cat([W_if, W_hf], dim=2).view(hidden_size, -1)
        b_f = b_if + b_hf if self.bias else None
        x_f = F.linear(x_f, W_f, b_f)
        f = torch.sigmoid(x_f)

        # (B, input_size + hidden_size)
        x_g = torch.cat([dropout(input), dropout(h_0)], dim=1)
        # (hidden_size, input_size + hidden_size)
        W_g = torch.cat([W_ig, W_hg], dim=2).view(hidden_size, -1)
        b_g = b_ig + b_hg if self.bias else None
        x_g = F.linear(x_g, W_g, b_g)
        g = torch.tanh(x_g)

        # (B, input_size + hidden_size)
        x_o = torch.cat([dropout(input), dropout(h_0)], dim=1)
        # (hidden_size, input_size + hidden_size)
        W_o = torch.cat([W_io, W_ho], dim=2).view(hidden_size, -1)
        b_o = b_io + b_ho if self.bias else None
        x_o = F.linear(x_o, W_o, b_o)
        o = torch.sigmoid(x_o)

        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)

        return h_1, c_1
