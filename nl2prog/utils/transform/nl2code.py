import torch
import numpy as np
from typing import Callable, List, Any, Optional, Tuple
from nl2prog.language.action import ActionSequence, ActionOptions
from nl2prog.language.evaluator import Evaluator
from nl2prog.encoders import ActionSequenceEncoder


class TransformCode:
    def __init__(self,
                 to_action_sequence: Callable[[Any],
                                              Optional[ActionSequence]],
                 action_sequence_encoder: ActionSequenceEncoder,
                 options: ActionOptions = ActionOptions(True, True)):
        self.to_action_sequence = to_action_sequence
        self.action_sequence_encoder = action_sequence_encoder
        self.options = options

    def __call__(self, code: Any, query_for_synth: List[str]) \
            -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        action_sequence = self.to_action_sequence(code)
        if action_sequence is None:
            return None
        evaluator = Evaluator(options=self.options)
        for action in action_sequence:
            evaluator.eval(action)
        a = self.action_sequence_encoder.encode_action(evaluator,
                                                       query_for_synth)
        p = self.action_sequence_encoder.encode_parent(evaluator)
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        action_tensor = torch.cat(
            [a[1:-1, 0].view(-1, 1), p[1:-1, 1:3].view(-1, 2)],
            dim=1)
        prev_action = a[:-2, 1:]
        return action_tensor, prev_action
