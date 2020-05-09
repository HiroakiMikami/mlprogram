import torch
import torch.nn as nn


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True,
                 padding_mode: str = "zeros"):
        super(SeparableConv1d, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation, groups=in_channels,
                                        bias=False, padding_mode=padding_mode)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1,
                                        bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input: torch.Tensor
            (N, in_channels, L) where N is the batch size, L is the sequence
            length.

        Returns
        -------
        output: torch.Tensor
            (N, out_channels, L) where N is the batch size, L is the sequence
            length.
        """
        return self.pointwise_conv(self.depthwise_conv(input))
