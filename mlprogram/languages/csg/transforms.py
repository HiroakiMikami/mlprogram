from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from mlprogram.languages.csg import AST, Interpreter


def per_canvas(variable: List[np.array]) -> torch.Tensor:
    return torch.stack([torch.tensor(canvas).unsqueeze(0).float() - 0.5
                        for canvas in variable], dim=0)


class TransformInputs(nn.Module):
    def forward(self, test_cases: List[Tuple[None, np.array]]) -> torch.Tensor:
        outputs = [output for _, output in test_cases]
        out = per_canvas(outputs)
        return out


class TransformVariables(nn.Module):
    def forward(self, variables: List[np.array],
                test_case_tensor: torch.Tensor) -> torch.Tensor:
        s = test_case_tensor.shape  # (N, C)
        if len(variables) == 0:
            return torch.zeros((0, *s))
        else:
            return torch.stack([
                per_canvas(canvas) for canvas in variables
            ])


class AddTestCases(nn.Module):
    def __init__(self, interpreter: Interpreter):
        super().__init__()
        self.interpreter: Interpreter = interpreter

    def __call__(self, ground_truth: AST) -> List[Tuple[None, np.array]]:
        return[
            (None, output)
            for output in self.interpreter.eval(ground_truth, [None])]
