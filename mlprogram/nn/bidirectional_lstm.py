import torch
import torch.nn as nn

from mlprogram.nn.embedding import EmbeddingWithMask
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class BidirectionalLSTM(nn.Module):
    def __init__(self, n_elem: int, embedding_size: int, hidden_size: int,
                 dropout: float = 0.0):
        """
        Parameters
        ----------
        n_elem: int
            The number of words including <unknown> label.
        embedding_size: int
            The dimention of each embedding
        hidden_size: int
            The number of features in LSTM
        dropout: float
            The probability of dropout
        """
        super().__init__()
        assert(hidden_size % 2 == 0)
        self.hidden_size = hidden_size
        self._embedding = EmbeddingWithMask(n_elem, embedding_size, -1)
        self._forward_lstm = nn.LSTMCell(embedding_size, hidden_size // 2)
        self._backward_lstm = nn.LSTMCell(embedding_size, hidden_size // 2)
        self._dropout_in = nn.Dropout(dropout)
        self._dropout_h = nn.Dropout(dropout)

    def forward(self, x: PaddedSequenceWithMask) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        x: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The padding value should be -1.

        Returns
        -------
        y: rnn.PaddedSeqeunceWithMask
            The output sequences of the LSTM
        """
        # Embed query
        embeddings = self._embedding(x.data)  # (embedding_dim,)
        embeddings = rnn.PaddedSequenceWithMask(embeddings, x.mask)

        L, B, _ = embeddings.data.shape
        device = embeddings.data.device

        # forward
        output = []
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L):
            tmp = embeddings.data[i, :, :].view(B, -1)
            tmp = self._dropout_in(tmp)
            h = self._dropout_h(h)
            h, c = self._forward_lstm(tmp, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output.append(h)

        # backward
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L - 1, -1, -1):
            tmp = embeddings.data[i, :, :].view(B, -1)
            tmp = self._dropout_in(tmp)
            h = self._dropout_h(h)
            h, c = self._backward_lstm(tmp, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output[i] = torch.cat([output[i], h], dim=1) \
                .view(1, B, -1)  # (1, B, hidden_size)

        output = torch.cat(output, dim=0)  # (L, B, hidden_size)
        features = rnn.PaddedSequenceWithMask(output, x.mask)
        return features
