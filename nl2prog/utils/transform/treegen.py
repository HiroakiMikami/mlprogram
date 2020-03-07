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
                 max_arity: int,
                 options: ActionOptions = ActionOptions(True, True)):
        self.to_action_sequence = to_action_sequence
        self.action_sequence_encoder = action_sequence_encoder
        self.max_arity = max_arity
        self.options = options

    def __call__(self, code: Any, query_for_synth: List[str]) \
            -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor]]:
        action_sequence = self.to_action_sequence(code)
        if action_sequence is None:
            return None
        evaluator = Evaluator(options=self.options)
        for action in action_sequence:
            evaluator.eval(action)
        a = self.action_sequence_encoder.encode_action(evaluator,
                                                       query_for_synth)
        if a is None:
            return None
        if np.any(a[-1, :].numpy() != -1):
            return None
        prev_action = a[:-2, 1:]

        rule_prev_action = \
            self.action_sequence_encoder.encode_each_action(
                evaluator, query_for_synth, self.max_arity)
        rule_prev_action = rule_prev_action[:-1]

        depth, matrix = self.action_sequence_encoder.encode_tree(evaluator)
        depth = depth[:-1]
        matrix = matrix[:-1, :-1]

        return prev_action, rule_prev_action, depth, matrix
