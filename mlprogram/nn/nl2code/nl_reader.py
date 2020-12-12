from typing import cast

import torch
import torch.nn as nn

from mlprogram import Environment
from mlprogram.nn.embedding import EmbeddingWithMask
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class NLReader(nn.Module):
    def __init__(self, num_words: int, embedding_dim: int, hidden_size: int,
                 dropout: float = 0.0):
        """
        Parameters
        ----------
        num_words: int
            The number of words including <unknown> label.
        embedding_dim: int
            The dimention of each embedding
        hidden_size: int
            The number of features in LSTM
        dropout: float
            The probability of dropout
        """
        super(NLReader, self).__init__()
        assert(hidden_size % 2 == 0)
        self.num_words = num_words
        self.hidden_size = hidden_size
        self._embedding = EmbeddingWithMask(num_words, embedding_dim,
                                            num_words)
        self._forward_lstm = nn.LSTMCell(embedding_dim, hidden_size // 2)
        self._backward_lstm = nn.LSTMCell(embedding_dim, hidden_size // 2)
        self._dropout_in = nn.Dropout(dropout)
        self._dropout_h = nn.Dropout(dropout)

    def forward(self, inputs: Environment) -> Environment:
        """
        Parameters
        ----------
        word_nl_query: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The padding value should be -1.

        Returns
        -------
        word_nl_query_features: rnn.PaddedSeqeunceWithMask
            The output sequences of the LSTM
        """
        nl_query = cast(PaddedSequenceWithMask, inputs["word_nl_query"])
        # Embed query
        q = nl_query.data + (nl_query.data == -1).long() * (self.num_words + 1)
        embeddings = self._embedding(q)  # (embedding_dim,)
        embeddings = rnn.PaddedSequenceWithMask(embeddings, nl_query.mask)

        L, B, _ = embeddings.data.shape
        device = embeddings.data.device

        # forward
        output = []
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L):
            x = embeddings.data[i, :, :].view(B, -1)
            x = self._dropout_in(x)
            h = self._dropout_h(h)
            h, c = self._forward_lstm(x, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output.append(h)

        # backward
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L - 1, -1, -1):
            x = embeddings.data[i, :, :].view(B, -1)
            x = self._dropout_in(x)
            h = self._dropout_h(h)
            h, c = self._backward_lstm(x, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output[i] = torch.cat([output[i], h], dim=1) \
                .view(1, B, -1)  # (1, B, hidden_size)

        output = torch.cat(output, dim=0)  # (L, B, hidden_size)
        features = rnn.PaddedSequenceWithMask(output, nl_query.mask)
        inputs["nl_query_features"] = features
        inputs["reference_features"] = features
        return inputs
