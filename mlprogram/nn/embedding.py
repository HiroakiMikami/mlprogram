import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingWithMask(nn.Embedding):
    """
    The embedding layer masking invalid values as 0
    """

    def __init__(self, n_id: int, embedding_size: int,
                 ignore_id: int):
        """
        Constructor

        Parameters
        ----------
        n_id : int
            Size of the dictionary of embeddings
        embedding_size : int
            Size of each embedding vector
        ignore_id : int
            The ID that should be masked

        Returns
        -------
        nn.Module
            The emebedding layer
        """
        super().__init__(n_id, embedding_size)
        nn.init.uniform_(self.weight, -0.1, 0.1)
        self.ignore_id = ignore_id

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        y = torch.where(x == self.ignore_id, torch.zeros_like(x), x)
        embedding = super().forward(y)
        return torch.where((x == self.ignore_id).unsqueeze(-1),
                           torch.zeros_like(embedding),
                           embedding)


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
