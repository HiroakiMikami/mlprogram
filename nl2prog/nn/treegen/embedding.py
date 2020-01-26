import torch
from typing import Tuple
import torch.nn as nn

from nl2prog.nn import EmbeddingWithMask


class Embedding(nn.Module):
    def __init__(self, vocab_num: int, element_vocab_num: int,
                 max_seq_num: int,
                 embedding_dim: int, elem_embedding_dim: int,
                 elem_feature_dim: int, padding_value: int = 0):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_num, embedding_dim)
        self.elem_embed = EmbeddingWithMask(element_vocab_num,
                                            elem_embedding_dim, padding_value)
        self.elem_to_seq = nn.Conv1d(elem_embedding_dim, elem_feature_dim,
                                     max_seq_num, bias=False)

    def forward(self, sequence: torch.Tensor, element_seq: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sequence: torch.Tensor
            The shape is (L, N). where L is the sequence length and
            N is the batch size.
        element_seq: rnn.PaddedSequenceWithMask
            The shape is (L, N, max_seq_len). where L is the sequence length
            and N is the batch size.

        Returns
        -------
        seq_embed: torch.Tensor
            The shape is (L, N, embedding_dim). where L is the sequence length
            and N is the batch size.
        elem_seq_embed: torch.Tensor
            The shape is (L, N, elem_feature_dim). where L is the sequence
            length and N is the batch size.
        """
        L, N = sequence.shape[:2]
        n = element_seq.shape[2]
        seq_embed = self.embed(sequence)
        elem_seq_embed = self.elem_embed(element_seq)
        elem_seq_embed = self.elem_to_seq(
            elem_seq_embed.reshape(L * N, n, -1).permute(0, 2, 1)
        ).reshape(L, N, -1)
        return seq_embed, elem_seq_embed
