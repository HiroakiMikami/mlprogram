import torch
import torch.nn as nn


class PointerNetwork(nn.Module):
    def __init__(self, key_size: int, value_size: int, hidden_size: int):
        super(PointerNetwork, self).__init__()
        self.w1 = nn.Linear(key_size, hidden_size, bias=False)
        self.w2 = nn.Linear(value_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, key: torch.Tensor, value: torch.Tensor,
                value_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        key: torch.Tensor
            (N, key_size) where N is the batch size.
        value: torch.Tensor
            (Lv, N, value_size) where Lv is the sequence length, N is
            the batch size.
        value_mask: torch.LongTensor
            (Lv, N) where Lv is the sequence length, N is the batch size.

        Returns
        -------
        log_prob: torch.Tensor
            (Lv, N) where Lv is the sequence length, N is the batch size.
        """
        Lv, N, _ = value.shape
        key_trans = self.w1(key).view(1, N, -1)
        value_trans = self.w2(value.view(Lv * N, -1)).view(Lv, N, -1)
        trans = torch.tanh(key_trans + value_trans)
        xi = self.v(trans.view(Lv * N, -1)).view(Lv, N)
        return xi * value_mask.to(xi.dtype) - \
            torch.log(torch.sum(torch.exp(xi) * value_mask.to(xi.dtype),
                                dim=0, keepdim=True))
