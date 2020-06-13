from typing import Union
from torch import Tensor as TorchTensor
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask

Tensor = Union[TorchTensor, PaddedSequenceWithMask]
