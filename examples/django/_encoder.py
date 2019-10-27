import torch
import torch.nn as nn
from nl2code.nn import LSTMCell
import nl2code.nn.utils.rnn as rnn


class Encoder(nn.Module):
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
        super(Encoder, self).__init__()
        assert(hidden_size % 2 == 0)
        self.hidden_size = hidden_size
        self._embedding = nn.Embedding(num_words, embedding_dim)
        self._forward_lstm = LSTMCell(embedding_dim, hidden_size // 2,
                                      dropout=dropout)
        self._backward_lstm = LSTMCell(embedding_dim, hidden_size // 2,
                                       dropout=dropout)

    def forward(self, query: rnn.PaddedSequenceWithMask) \
            -> rnn.PaddedSequenceWithMask:
        """
        Parameters
        ----------
        query: rnn.PaddedSequenceWithMask
            The minibatch of sequences.
            The shape of each sequence is (sequence_length).

        Returns
        -------
        output: rnn.PaddedSeqeunceWithMask
            The output sequences of the LSTM
        """
        # Embed query
        embeddings = self._embedding(query.data)  # (embedding_dim,)
        embeddings = rnn.PaddedSequenceWithMask(embeddings, query.mask)

        L, B, _ = embeddings.data.shape
        device = embeddings.data.device

        # forward
        output = []
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L):
            x = embeddings.data[i, :, :].view(B, -1)
            h, c = self._forward_lstm(x, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output.append(h)

        # backward
        h = torch.zeros(B, self.hidden_size // 2, device=device)
        c = torch.zeros(B, self.hidden_size // 2, device=device)
        for i in range(L - 1, -1, -1):
            x = embeddings.data[i, :, :].view(B, -1)
            h, c = self._backward_lstm(x, (h, c))
            h = h * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            c = c * embeddings.mask[i, :].view(B, -1)  # (B, hidden_size // 2)
            output[i] = torch.cat([output[i], h], dim=1) \
                .view(1, B, -1)  # (1, B, hidden_size)

        output = torch.cat(output, dim=0)  # (L, B, hidden_size)
        return rnn.PaddedSequenceWithMask(output, query.mask)
