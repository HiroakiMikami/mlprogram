import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn._VF as _VF
from typing import Tuple

from nl2code.nn.utils import rnn


def query_history(history: torch.FloatTensor, index: torch.LongTensor):
    """
    Return the hidden states of the specified indexes

    Parameters
    ----------
    history : torch.FloatTensor
        The sequence of history. The shape is (L, B, F)
    index : torch.LongTensor
        The indexes of history to query. The shape is (B,)

    Returns
    -------
    torch.FloatTensor
        The hidden states of the specified indexes. The shape is (B, F)
    """

    device = history.device
    L = history.shape[0]
    B = history.shape[1]
    index_onehot = torch.eye(L, device=device)[index]  # (B, L)
    index_onehot = index_onehot.reshape((B, 1, L))  # (B, 1, L)
    h = torch.matmul(index_onehot, history.permute([1, 0, 2]))  # (B, 1, *)
    h = h.reshape((B, *history.shape[2:]))  # (B, *)
    return h


class DecoderCell(nn.Module):
    """
    One cell of the decoder

    Notes
    -----
    The LSTM cell initialization and dropout are slightly different from
    the original implementation.
    """

    def __init__(self, query_size: int, input_size: int, hidden_size: int,
                 att_hidden_size: int, dropout: float = 0.0):
        """
        Constructor

        Parameters
        ----------
        query_size: int
            Size of each query vector
        input_size: int
            Size of each input vector
        hidden_size: int
            The number of features in the hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        dropout: float
            The probability of dropout
        """
        super(DecoderCell, self).__init__()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size))
        self.weight_ch = nn.Parameter(
            torch.Tensor(4 * hidden_size, query_size))
        self.weight_ph = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self._attention_layer1 = nn.Linear(
            query_size + hidden_size, att_hidden_size)
        self._attention_layer2 = nn.Linear(att_hidden_size, 1)

        nn.init.orthogonal_(self.weight_hh)
        nn.init.orthogonal_(self.weight_ch)
        nn.init.orthogonal_(self.weight_ph)
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.zeros_(self.bias_hh)
        nn.init.zeros_(self.bias_ih)
        self.bias_ih.data[hidden_size:(2 * hidden_size)] = 1
        nn.init.xavier_uniform_(self._attention_layer1.weight)
        nn.init.zeros_(self._attention_layer1.bias)
        nn.init.xavier_uniform_(self._attention_layer2.weight)
        nn.init.zeros_(self._attention_layer2.bias)

    def forward(self,
                query: rnn.PaddedSequenceWithMask,
                input: torch.FloatTensor,
                parent_index: torch.LongTensor, history: torch.FloatTensor,
                state: Tuple[torch.FloatTensor, torch.FloatTensor]
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Parameters
        ----------
        query: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        input: torch.FloatTensor
            The input feature vector. The shape is (B, input_size)
        parent_index: torch.LongTensor
            The indexes of the parent actions. The shape is (B).
            If index is -1, it means that the action is root action.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        (h_0, c_0): Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the LSTM initial states. The shape of each tensor is
            (B, hidden_size)

        Returns
        -------
        ctx_vec : torch.FloatTensor
            The context vector of this cell. The shape is (B, query_size)
        (h_1, c_1) : Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the next states. The shape of each tensor is
            (B, hidden_size)
        """
        h_0, c_0 = state
        L_q, B, query_size = query.data.shape
        _, hidden_size = h_0.shape
        L_h, _, _ = history.shape

        # Context
        h_context = h_0.reshape([1, B, hidden_size])
        h_context = h_context.expand(L_q, B, hidden_size)
        # (L_q, B, query_size + hidden_size)
        att = torch.cat([h_context, query.data], dim=2)
        # (L_q, B, att_hidden_size)
        att_hidden = torch.tanh(self._attention_layer1(att))
        att_raw = self._attention_layer2(att_hidden)  # (L_q, B, 1)
        att_raw = att_raw.reshape([L_q, B])  # (L_q, B)
        ctx_att = torch.exp(att_raw -
                            torch.max(att_raw, dim=0, keepdim=True).values
                            )  # (L_q, B)
        ctx_att = ctx_att * query.mask.float()  # (L_q, B)
        ctx_att = ctx_att / torch.sum(ctx_att, dim=0, keepdim=True)  # (L_q, B)
        ctx_att = ctx_att.reshape([L_q, B, 1])  # (L_q, B, 1)
        ctx_vec = torch.sum(query.data * ctx_att, dim=0)  # (B, query_size)

        # Parent_history
        device = history.device
        h_root = torch.zeros(1, B, hidden_size, device=device)
        history = torch.cat([h_root, history], dim=0)
        h_parent = query_history(history, parent_index + 1)

        if self.training:
            weight_ih = self.weight_ih.view(4, hidden_size, -1)
            W_ii, W_if, W_ig, W_io = torch.split(weight_ih, 1, dim=0)
            weight_ch = self.weight_ch.view(4, hidden_size, -1)
            W_ci, W_cf, W_cg, W_co = torch.split(weight_ch, 1, dim=0)
            weight_ph = self.weight_ph.view(4, hidden_size, -1)
            W_pi, W_pf, W_pg, W_po = torch.split(weight_ph, 1, dim=0)
            weight_hh = self.weight_hh.view(4, hidden_size, -1)
            W_hi, W_hf, W_hg, W_ho = torch.split(weight_hh, 1, dim=0)
            bias_ih = self.bias_ih.view(4, -1)
            b_ii, b_if, b_ig, b_io = torch.split(bias_ih, 1, dim=0)
            bias_hh = self.bias_hh.view(4, -1)
            b_hi, b_hf, b_hg, b_ho = torch.split(bias_hh, 1, dim=0)
            dropout = self.dropout

            # (B, input_size+query_size+hidden_size+hidden_size)
            x_i = torch.cat([dropout(input), ctx_vec, h_parent, dropout(h_0)],
                            dim=1)
            W_i = torch.cat([W_ii, W_ci, W_pi, W_hi], dim=2) \
                .view(hidden_size, -1)
            b_i = b_ii + b_hi
            x_i = F.linear(x_i, W_i, b_i)
            i = torch.sigmoid(x_i)

            x_f = torch.cat([dropout(input), ctx_vec, h_parent, dropout(h_0)],
                            dim=1)
            W_f = torch.cat([W_if, W_cf, W_pf, W_hf], dim=2) \
                .view(hidden_size, -1)
            b_f = b_if + b_hf
            x_f = F.linear(x_f, W_f, b_f)
            f = torch.sigmoid(x_f)

            x_g = torch.cat([dropout(input), ctx_vec, h_parent, dropout(h_0)],
                            dim=1)
            W_g = torch.cat([W_ig, W_cg, W_pg, W_hg], dim=2) \
                .view(hidden_size, -1)
            b_g = b_ig + b_hg
            x_g = F.linear(x_g, W_g, b_g)
            g = torch.tanh(x_g)

            x_o = torch.cat([dropout(input), ctx_vec, h_parent, dropout(h_0)],
                            dim=1)
            W_o = torch.cat([W_io, W_co, W_po, W_ho], dim=2) \
                .view(hidden_size, -1)
            b_o = b_io + b_ho
            x_o = F.linear(x_o, W_o, b_o)
            o = torch.sigmoid(x_o)

            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            # (B, input_size+query_size+hidden_size)
            x = torch.cat([input, ctx_vec, h_parent], dim=1)
            weight_ih = self.weight_ih.view(4, hidden_size, -1)
            W_ii, W_if, W_ig, W_io = torch.split(weight_ih, 1, dim=0)
            weight_ch = self.weight_ch.view(4, hidden_size, -1)
            W_ci, W_cf, W_cg, W_co = torch.split(weight_ch, 1, dim=0)
            weight_ph = self.weight_ph.view(4, hidden_size, -1)
            W_pi, W_pf, W_pg, W_po = torch.split(weight_ph, 1, dim=0)
            W_i = torch.cat([W_ii, W_ci, W_pi], dim=2).view(hidden_size, -1)
            W_f = torch.cat([W_if, W_cf, W_pf], dim=2).view(hidden_size, -1)
            W_g = torch.cat([W_ig, W_cg, W_pg], dim=2).view(hidden_size, -1)
            W_o = torch.cat([W_io, W_co, W_po], dim=2).view(hidden_size, -1)
            W = torch.cat([W_i, W_f, W_g, W_o], dim=0)
            h_1, c_1 = _VF.lstm_cell(x, (h_0, c_0), W,
                                     self.weight_hh, self.bias_ih,
                                     self.bias_hh)

        return ctx_vec, (h_1, c_1)


class Decoder(nn.Module):
    def __init__(self, query_size: int, input_size: int, hidden_size: int,
                 att_hidden_size: int, dropout: float = 0.0):
        """
        Constructor

        Parameters
        ----------
        query_size: int
            Size of each query vector
        input_size: int
            Size of each input vector
        hidden_size: int
            The number of features in the hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        dropout: float
            The probability of dropout
        """
        super(Decoder, self).__init__()
        self._cell = DecoderCell(
            query_size, input_size, hidden_size, att_hidden_size,
            dropout=dropout)

    def forward(self,
                query: rnn.PaddedSequenceWithMask,
                input: torch.nn.utils.rnn.PackedSequence,
                parent_index: torch.nn.utils.rnn.PackedSequence,
                history: torch.FloatTensor,
                state: Tuple[torch.FloatTensor, torch.FloatTensor]
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                           Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Parameters
        ----------
        query: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        input: rnn.PackedSequenceWithMask
            The input sequence of feature vectors.
            The shape of input is (L_a, B, input_size)
        parent_index: rnn.PackedSequenceWithMask
            The sequence of parent action indexes.
            If index is 0, it means that the action is root action.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (B, L_h, hidden_size)
        (h_0, c_0): Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the LSTM initial states. The shape of each tensor is
            (B, hidden_size)

        Returns
        -------
        output: torch.nn.utils.rnn.PackedSequence
            Packed sequence containing the output hidden states.
        contexts: torch.nn.utils.rnn.PackedSequence
            Packed sequence containing the context vectors.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        (h_n, c_n) : Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the next states. The shape of each tensor is
            (B, hidden_size)
        """
        hs = []
        cs = []
        for d, i in zip(torch.split(input.data, 1, dim=0),
                        torch.split(parent_index.data, 1, dim=0)):
            ctx, state = self._cell(query, d.reshape(
                d.shape[1:]), i, history, state)
            hs.append(state[0])
            cs.append(ctx)
            history = torch.cat([history,
                                 state[0].reshape(1, *state[0].shape)], dim=0)
        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return (rnn.PaddedSequenceWithMask(hs, input.mask),
                rnn.PaddedSequenceWithMask(cs, input.mask),
                history,
                state)
