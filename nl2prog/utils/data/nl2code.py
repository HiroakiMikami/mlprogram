import torch
from typing import List, Tuple

from nl2prog.nn.utils import rnn
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class CollateInput:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, inputs: List[torch.Tensor]) -> PaddedSequenceWithMask:
        inputs = rnn.pad_sequence(inputs, padding_value=-1)

        return inputs.to(self.device)


class CollateActionSequence:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, action_sequence: List[Tuple[torch.Tensor,
                                                   torch.Tensor]]) \
            -> Tuple[PaddedSequenceWithMask, PaddedSequenceWithMask]:
        actions = []
        prev_actions = []
        for action, prev_action in action_sequence:
            actions.append(action)
            prev_actions.append(prev_action)
        actions = rnn.pad_sequence(actions, padding_value=-1)
        prev_actions = rnn.pad_sequence(prev_actions, padding_value=-1)

        return (actions.to(self.device), prev_actions.to(self.device))


class CollateState:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, states: List[Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(states) == 0 or states[0] is None:
            return None
        hist = \
            torch.stack([state[0] for state in states], dim=1).to(self.device)
        h_n = \
            torch.stack([state[1] for state in states], dim=0).to(self.device)
        c_n = \
            torch.stack([state[2] for state in states], dim=0).to(self.device)
        return hist, h_n, c_n


def split_states(state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
        -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    history, h_n, c_n = state
    state_size = history.shape[2]
    history = torch.split(history, 1, dim=1)
    history = [x.reshape(-1, state_size) for x in history]
    h_n = torch.split(h_n, 1, dim=0)
    h_n = [x.view(-1) for x in h_n]
    c_n = torch.split(c_n, 1, dim=0)
    c_n = [x.view(-1) for x in c_n]
    return list(zip(history, h_n, c_n))
