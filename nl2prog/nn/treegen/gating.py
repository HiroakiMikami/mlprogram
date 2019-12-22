import torch
import torch.nn as nn


class Gating(nn.Module):
    def __init__(self, in0_size: int, in1_size: int, query_size: int,
                 hidden_size: int):
        super(Gating, self).__init__()
        self.q = nn.Linear(in0_size, query_size, bias=False)
        self.w_k0 = nn.Linear(in0_size, query_size, bias=False)
        self.w_k1 = nn.Linear(in1_size, query_size, bias=False)
        self.w_f0 = nn.Linear(in0_size, hidden_size, bias=False)
        self.w_f1 = nn.Linear(in1_size, hidden_size, bias=False)

    def forward(self, input0: torch.Tensor,
                input1: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input0: torch.Tensor
            (L, N, in0_size) where L is the sequence length, N is
            the batch size.
        input1: torch.Tensor
            (L, N, in1_size) where L is the sequence length, N is
            the batch size.

        Returns
        -------
        output: torch.Tensor
            (L, N, hidden_size) where L is the sequence length, N is
            the batch size.
        """
        L, N, _ = input0.shape
        q = self.q(input0.view(L * N, -1)).view(L, N, -1)  # (L, N, E)
        k0 = self.w_k0(input0.view(L * N, -1)).view(L, N, -1)  # (L, N, E)
        k1 = self.w_k1(input1.view(L * N, -1)).view(L, N, -1)  # (L, N, E)
        # (L, N, 2)
        alpha = torch.cat([torch.bmm(q.view(L * N, 1, -1),
                                     k0.view(L * N, -1, 1)),
                           torch.bmm(q.view(L * N, 1, -1),
                                     k1.view(L * N, -1, 1))]).view(L, N, -1)
        alpha = torch.softmax(alpha, dim=2)  # (L, N, 2)
        v0 = self.w_f0(input0.view(L * N, -1)).view(L, N, -1)
        v1 = self.w_f1(input1.view(L * N, -1)).view(L, N, -1)
        return alpha[:, :, 0:1] * v0 + alpha[:, :, 1:2] * v1
