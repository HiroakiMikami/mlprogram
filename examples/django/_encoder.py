import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, num_words: int, embedding_dim: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        """
        Parameters
        ----------
        num_words: int
            The number of words including <unknown> label.
        embedding_dim: int
            The dimention of each embedding
        hidden_state: int
            The number of features in LSTM
        num_layers: int
            The number of recurrent layers
        dropout: float
            The probability of dropout
        """
        super(Encoder, self).__init__()
        assert(hidden_size % 2 == 0)
        self._embedding = nn.Embedding(num_words, embedding_dim)
        self._lstm = nn.LSTM(embedding_dim, hidden_size // 2,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=True)

    def forward(self, query: rnn.PackedSequence) \
            -> Tuple[rnn.PackedSequence, Tuple[torch.FloatTensor,
                                               torch.FloatTensor]]:
        """
        Parameters
        ----------
        query: rnn.PackedSequence
            The minibatch of sequences.
            The shape of each sequence is (sequence_length).

        Returns
        -------
        output: rnn.PackedSequence
            The output sequences of the LSTM
        h_n, c_n: Tuple[torch.FloatTensor, torch.FloatTensor]
            The final states of the LSTM
        """
        # Embed query
        embeddings = self._embedding(query.data)  # (embedding_dim,)
        embeddings = rnn.PackedSequence(embeddings, query.batch_sizes,
                                        query.sorted_indices,
                                        query.unsorted_indices)
        return self._lstm(embeddings)
