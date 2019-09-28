import torch
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class PaddedSequenceWithMask:
    data: torch.FloatTensor
    mask: torch.LongTensor


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
    max_length = data.shape[0]
    B = data.shape[1]
    mask_shape = np.ones((len(data.shape),))
    mask_shape[1] = data.shape[1]

    indexes = torch.arange(max_length).long().reshape(
        [max_length, 1]).expand([max_length, B])
    mask = lengths.unsqueeze(1).expand(
        [B, max_length]).permute([1, 0])  # [B, max_length]
    mask = (indexes < mask).long()
    return PaddedSequenceWithMask(data, mask)
