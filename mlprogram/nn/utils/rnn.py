import torch
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class PaddedSequenceWithMask:
    data: torch.FloatTensor
    mask: torch.LongTensor

    def to(self, *args, **kwargs):
        return PaddedSequenceWithMask(self.data.to(*args, **kwargs),
                                      self.mask.to(*args, **kwargs))

    def cuda(self):
        return PaddedSequenceWithMask(self.data.cuda(), self.mask.cuda())


def pad_sequence(sequences: List[torch.FloatTensor],
                 padding_value: float = 0.0) -> PaddedSequenceWithMask:
    """
    Pad a list of variable length Tensors and create its mask tensor.

    Parameters
    ----------
    sequences : List[torch.FloatTensor]
        The list of variable length Tensors
    padding_value: float

    Returns
    -------
    PaddedSequenceWithMask
        The padded tensor and its mask tensor
    """
    data = torch.nn.utils.rnn.pad_sequence(sequences,
                                           padding_value=padding_value)
    L, B = data.shape[:2]
    mask = torch.zeros(L, B, device=data.device)
    for i in range(B):
        mask[:len(sequences[i]), i] = 1
    return PaddedSequenceWithMask(data, mask)
    return pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False),
        padding_value=padding_value)


def pad_packed_sequence(sequence: torch.nn.utils.rnn.PackedSequence,
                        padding_value: float = 0.0) -> PaddedSequenceWithMask:
    """
    Pad a packed sequence and create its mask tensor.

    Parameters
    ----------
    sequences : torch.nn.utils.rnn.PackedSequence
        The packed sequence of the variable Tensors
    padding_value: float

    Returns
    -------
    PaddedSequenceWithMask
        The padded tensor and its mask tensor
    """
    data, lengths = torch.nn.utils.rnn.pad_packed_sequence(
        sequence, padding_value=padding_value)  # [T, B, x], [B]
    lengths = lengths.to(data.device)
    max_length = data.shape[0]
    B = data.shape[1]
    mask_shape = np.ones((len(data.shape),))
    mask_shape[1] = data.shape[1]

    indexes = torch.arange(max_length, device=data.device).long().reshape(
        [max_length, 1]).expand([max_length, B])
    mask = lengths.unsqueeze(1).expand(
        [B, max_length]).permute([1, 0])  # [B, max_length]
    mask = (indexes < mask).long()
    return PaddedSequenceWithMask(data, mask)
