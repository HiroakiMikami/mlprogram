from math import pi, sqrt

import torch


def gelu(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * input * \
        (1 + torch.tanh(sqrt(2 / pi) *
                        (input + 0.044715 * torch.pow(input, 3))))
