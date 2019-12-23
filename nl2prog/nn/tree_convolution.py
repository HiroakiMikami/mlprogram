import torch
import torch.nn as nn

from nl2prog.nn.functional import bmm


class TreeConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 bias: bool = True):
        super(TreeConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(kernel_size * in_channels, out_channels, 1,
                              bias=bias)

    def forward(self, input: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor
            (N, in_channels, L) where N is the batch size, L is the sequence
            length.
        m: torch.Tensor
            (N, L, L) where N is the batch size, L is the sequence
            length. This represents the adjacency matrix of the tree.
            This tensor can be sparse tensor.

        Returns
        -------
        output: torch.Tensor
            (N, out_channels, L) where N is the batch size, L is the sequence
            length.
        """
        inputs = [input]
        y = input
        for i in range(self.kernel_size - 1):
            y = bmm(y, m)
            inputs.append(y)

        # (N, kernel_size * in_channels, L)
        inputs = torch.cat(inputs, dim=1)
        return self.conv(inputs)
