import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingWithMask(nn.Embedding):
    """
    The embedding layer masking invalid values as 0
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 value_to_mask: int):
        """
        Constructor

        Parameters
        ----------
        num_embeddings : int
            Size of the dictionary of embeddings
        embedding_dim : int
            Size of each embedding vector
        value_to_mask : int
            The ID that should be masked

        Returns
        -------
        nn.Module
            The emebedding layer
        """
        super(EmbeddingWithMask, self).__init__(
            num_embeddings + 1, embedding_dim)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        self._value_to_mask = value_to_mask

    def forward(self, input: torch.LongTensor) -> torch.FloatTensor:
        embedding = super(EmbeddingWithMask, self).forward(input)
        mask = 1 - (input == self._value_to_mask).float()

        return embedding * mask.reshape([*mask.shape, 1])


class EmbeddingInverse(nn.Module):
    def __init__(self, num_embeddings: int, bias: bool = True):
        """
        Parameters
        ----------
        num_embeddings: int
            Size of the dictionary of embeddings
        bias: bool
            If se to `False`, the layer will not learn an additive bias.
        """
        super(EmbeddingInverse, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_embeddings))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, embedded: torch.FloatTensor, embedding: nn.Embedding) \
            -> torch.FloatTensor:
        """
        Parameters
        ----------
        embedded: torch.FloatTensor
            The embedded vector. The shape of
            (*, self._embedding.num_embeddings)
        embedding: nn.Embedding
            The embedding layer
        """
        return F.linear(embedded, embedding.weight, self.bias)
