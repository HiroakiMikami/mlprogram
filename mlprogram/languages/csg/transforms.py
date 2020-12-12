from typing import List

import numpy as np
import torch

from mlprogram import Environment
from mlprogram.languages.csg import Interpreter


class TransformCanvas:
    def per_canvas(self, variable: List[np.array]) -> torch.Tensor:
        return torch.stack([torch.tensor(canvas).unsqueeze(0).float() - 0.5
                            for canvas in variable], dim=0)

    def __call__(self, entry: Environment) -> Environment:
        if "test_cases" in entry:
            test_cases = entry["test_cases"]
            outputs = [output for _, output in test_cases]
            entry["test_case_tensor"] = self.per_canvas(outputs)
        if "variables" in entry:
            variables = entry["variables"]
            s = entry["test_case_tensor"].shape  # (N, C)
            if len(variables) == 0:
                entry["variables_tensor"] = torch.zeros((0, *s))
            else:
                entry["variables_tensor"] = torch.stack([
                    self.per_canvas(canvas) for canvas in variables
                ])
        return entry


class AddTestCases:
    def __init__(self, interpreter: Interpreter):
        self.interpreter = interpreter

    def __call__(self, entry: Environment) -> Environment:
        ground_truth = entry["ground_truth"]
        entry["test_cases"] = [
            (None, output)
            for output in self.interpreter.eval(ground_truth, [None])]
        return entry
