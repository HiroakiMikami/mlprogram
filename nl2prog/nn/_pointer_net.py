import torch
import torch.nn as nn

from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class PointerNet(nn.Module):
    def __init__(self, query_size: int, decoder_output_size: int,
                 hidden_size: int):
        """
        Parameters
        ----------
        query_size: int
            The size of each query embedding vector
        decoder_output_size: int
            The size of each decoder hidden state
        hidden_size: int
            The size of each hidden state.
        """
        super(PointerNet, self).__init__()

        self._l1_q = nn.Linear(query_size, hidden_size)
        self._l1_h = nn.Linear(decoder_output_size, hidden_size)
        self._l2 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self._l1_q.weight)
        nn.init.zeros_(self._l1_q.bias)
        nn.init.xavier_uniform_(self._l1_h.weight)
        nn.init.zeros_(self._l1_h.bias)
        nn.init.xavier_uniform_(self._l2.weight)
        nn.init.zeros_(self._l2.bias)

    def forward(self, query: PaddedSequenceWithMask,
                decoder_states: PaddedSequenceWithMask):
        """
        Parameters
        ----------
        query: PaddedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        decoder_states: PaddedSequenceWithMask
            The outputs of the decoder.
            The shape of the vector is (L_a, B, decoder_output_size)

        Returns
        -------
        PaddedSequenceWithMask
            The output of the pointer net
            The shape of the vector is (L_a, B, L_q)
        """
        L_q, B, _ = query.data.shape
        L_a, _, _ = decoder_states.data.shape

        query_trans = self._l1_q(query.data)  # (L_q, B, hidden_num)
        decoder_states_trans = self._l1_h(
            decoder_states.data)  # (L_a, B, hidden_num)

        _, _, hidden_num = query_trans.shape
        query_trans = query_trans.reshape(
            [1, L_q, B, hidden_num]).expand([L_a, L_q, B, hidden_num])
        decoder_states_trans = decoder_states_trans.reshape(
            [L_a, 1, B, hidden_num]).expand([L_a, L_q, B, hidden_num])

        # (L_a, L_q, B, hidden_num)
        trans = torch.tanh(query_trans + decoder_states_trans)
        scores = self._l2(trans).reshape([L_a, L_q, B])  # (L_a, L_q, B)
        scores = torch.exp(
            scores - torch.max(scores, dim=1, keepdim=True).values)
        mask = query.mask.reshape([1, L_q, B]).expand(
            [L_a, L_q, B])  # (L_a, L_q, B)
        scores = scores * mask.float()
        scores = scores / torch.sum(scores, dim=1, keepdim=True)

        scores = scores.permute([0, 2, 1])  # (L_a, B, L_q)
        return PaddedSequenceWithMask(scores, decoder_states.mask)
