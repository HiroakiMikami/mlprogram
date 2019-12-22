import torch


def lne_to_nel(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert data layout. This conversion is required to use convolution layer.

    Parameters
    ----------
    tensor: torch.Tensor
        (L, N, E) where L is the sequence length, N is the batch size, E is
        the embedding dimension.

    Returns
    -------
    tensor: torch.Tensor
        (N, E, L) where N is the batch size, E is the embedding dimension,
        L is the sequence length.
    """
    return tensor.permute(1, 2, 0)


def nel_to_lne(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert data layout. This conversion is required to use convolution layer.

    Parameters
    ----------
    tensor: torch.Tensor
        (N, E, L) where N is the batch size, E is the embedding dimension,
        L is the sequence length.

    Returns
    -------
    tensor: torch.Tensor
        (L, N, E) where L is the sequence length, N is the batch size, E is
        the embedding dimension.
    """
    return tensor.permute(2, 0, 1)
