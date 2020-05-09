import torch


def position_embeddings(position_tensor: torch.LongTensor, b: int, E: int,
                        dtype: torch.dtype = torch.float) \
        -> torch.Tensor:
    """
    Returns the embeddings for bth Transformer block

    Parameters
    ----------
    position_tensor: torch.LongTensor
        (L, N) where L is the sequence length, N is the batch size.
    b: int
        The index of Transformer block
    E: int
        The embedding dimension
    dtype: torch.dtype

    Returns
    -------
    embeddings: torch.Tensor
        (L, N, E) where L is the sequence length, E is the embedding dimension.
        embeddings[i, n, 2j    ] =
            sin((position_tensor[i, n] + b) / (10000**(2j/E)))
        embeddings[i, n, 2j + 1] =
            sin((position_tensor[i, n] + b) / (10000**(2j/E)))
    """
    device = position_tensor.device
    L, N = position_tensor.shape
    divisor = torch.arange(0, E) // 2
    divisor = \
        torch.pow(10000, 2 * divisor.to(dtype=dtype) / E).to(device=device)
    embeddings = \
        position_tensor.view(L, N, 1).expand(L, N, E)\
        .to(dtype=dtype).to(device=device)
    embeddings = embeddings + b
    embeddings = embeddings
    embeddings /= divisor
    return torch.sin(embeddings)


def index_embeddings(tensor: torch.Tensor, b: int) \
        -> torch.Tensor:
    """
    Returns the position embeddings for bth Transformer block

    Parameters
    ----------
    tensor: torch.Tensor
        (L, N, E) where L is the sequence length, N is the batch size, E is
        the embedding dimension.
    b: int
        The index of Transformer block
    Returns
    -------
    embeddings: torch.Tensor
        (L, E) where L is the sequence length, E is the embedding dimension.
        embeddings[i, 2j    ] = sin((i + b) / (10000**(2j/E)))
        embeddings[i, 2j + 1] = sin((i + b) / (10000**(2j/E)))
    """
    L, N, E = tensor.shape
    indexes = torch.arange(L, device=tensor.device).view(L, 1)
    return position_embeddings(indexes, b, E, tensor.dtype)
