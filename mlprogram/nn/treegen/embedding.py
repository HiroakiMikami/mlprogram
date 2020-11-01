import numpy as np
import torch
import torch.nn as nn

from mlprogram.nn import EmbeddingWithMask


class ElementEmbedding(nn.Module):
    def __init__(self, embed: nn.Module, max_elem_num: int,
                 elemwise_embedding_dim: int,
                 embedding_dim: int):
        super(ElementEmbedding, self).__init__()
        self.embed = embed
        self.elem_to_seq = nn.Conv1d(elemwise_embedding_dim, embedding_dim,
                                     max_elem_num, bias=False)

    def forward(self, input) -> torch.Tensor:
        embed = self.embed(input)
        dims = embed.shape
        embed = embed.reshape(np.prod(dims[:-2]), dims[-2], dims[-1]) \
            .permute(0, 2, 1)
        return self.elem_to_seq(embed).reshape(*dims[:-2], -1)


class ActionEmbedding(nn.Module):
    def __init__(self, rule_num: int, token_num: int, embedding_dim: int):
        super(ActionEmbedding, self).__init__()
        self.rule_num = rule_num
        self.token_num = token_num
        self.rule_embed = EmbeddingWithMask(rule_num, embedding_dim, rule_num)
        self.token_embed = EmbeddingWithMask(token_num + 1, embedding_dim,
                                             token_num + 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence: torch.Tensor
            The shape is (L, N, 3). where L is the sequence length and
            N is the batch size. The padding value should be -1.
            [:, :, 0] represent the rule IDs, [:, :, 1] represent the token
            IDs, [:, :, 2] represent the indexes of the queries.

        Returns
        -------
        seq_embed: torch.Tensor
            The shape is (L, N, embedding_dim). where L is the sequence length
            and N is the batch size.
        """
        L, N = sequence.shape[:2]

        rule_seq = sequence[:, :, 0]
        rule_seq = rule_seq + (rule_seq == -1) * (self.rule_num + 1)

        token_seq = sequence[:, :, 1]
        reference_seq = (token_seq == -1) * (sequence[:, :, 2] != -1)
        # reference_seq => self.token_num
        token_seq = token_seq + reference_seq * (self.token_num + 1)
        token_seq = token_seq + (token_seq == -1) * (self.token_num + 2)

        return self.rule_embed(rule_seq) + self.token_embed(token_seq)


class ActionSignatureEmbedding(nn.Module):
    def __init__(self, token_num: int, node_type_num: int,
                 embedding_dim: int):
        super(ActionSignatureEmbedding, self).__init__()
        self.token_num = token_num
        self.node_type_num = node_type_num
        self.node_type_embed = EmbeddingWithMask(
            node_type_num, embedding_dim, node_type_num)
        self.token_embed = EmbeddingWithMask(
            token_num + 1, embedding_dim, token_num + 1)

    def forward(self, signature: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        signature: torch.Tensor
            The shape is (-1, 3)
            The padding value should be -1.
            [:, 0] represent the IDs of the node types, [:, 1]
            represent the token IDs, [:, 2] represent the indexes of the
            queries.

        Returns
        -------
        embed: torch.Tensor
            The shape is (-1, feature_dim).
        """
        dims = signature.shape
        signature = signature.reshape(-1, dims[-1])

        node_type_seq = signature[:, 0]
        node_type_seq = node_type_seq + \
            (node_type_seq == -1) * (self.node_type_num + 1)
        token_seq = signature[:, 1]
        reference_seq = (token_seq == -1) * (signature[:, 2] != -1)
        token_seq = token_seq + reference_seq * (self.token_num + 2)
        token_seq = token_seq + \
            (token_seq == -1) * (self.token_num + 2)

        embed = self.node_type_embed(node_type_seq) + \
            self.token_embed(token_seq)
        return embed.reshape(*dims[:-1], -1)
