import torch
import torch.nn as nn

from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class PointerNet(nn.Module):
    def __init__(self, key_size: int, value_size: int, hidden_size: int,
                 bias: bool = True):
        super(PointerNet, self).__init__()
        self.w1 = nn.Linear(key_size, hidden_size, bias=bias)
        self.w2 = nn.Linear(value_size, hidden_size, bias=bias)
        self.v = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, key: torch.Tensor, value: PaddedSequenceWithMask) \
            -> torch.Tensor:
        """
        Parameters
        ----------
        key: torch.Tensor
            (Lk, N, key_size) where Lk is the key sequence length, N is
            the batch size.
        value: PaddedSequenceWithMask
            (Lv, N, value_size) where Lv is the value sequence length, N is
            the batch size.

        Returns
        -------
        torch.Tensor
            The output of the pointer net
            The shape of the vector is (Lk, B, Lv)
        """
        Lk, N, _ = key.shape
        Lv, _, _ = value.data.shape

        key_trans = self.w1(key)  # (Lk, N, hidden_num)
        value_trans = self.w2(value.data)  # (Lv, N, hidden_num)

        _, _, hidden_num = key_trans.shape
        key_trans = key_trans.reshape(
            [Lk, 1, N, hidden_num]).expand([Lk, Lv, N, hidden_num])
        value_trans = value_trans.reshape(
            [1, Lv, N, hidden_num]).expand([Lk, Lv, N, hidden_num])

        # (Lk, Lv, N, hidden_num)
        trans = torch.tanh(key_trans + value_trans)
        mask = value.mask.reshape([1, Lv, N]).expand(
            [Lk, Lv, N])  # (Lk, Lv, N)
        xi = self.v(trans).reshape([Lk, Lv, N])  # (Lk, Lv, N)
        scores = xi - \
            torch.log(torch.sum(torch.exp(xi) * mask.to(xi.dtype),
                                dim=1, keepdim=True))
        return scores.permute([0, 2, 1])  # (Lk, N, Lv)
