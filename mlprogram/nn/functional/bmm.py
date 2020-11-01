from typing import Union

import torch


def bmm(input: Union[torch.Tensor, torch.sparse_coo_tensor],
        mat2: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    input: torch.sparse_coo_tensor or torch.Tensor
    mat: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if input.is_sparse:
        batch_size = input.shape[0]
        return torch.stack([input[i].mm(mat2[i]) for i in range(batch_size)])
    else:
        return torch.bmm(input, mat2)
