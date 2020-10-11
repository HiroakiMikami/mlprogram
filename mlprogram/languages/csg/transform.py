import torch
import numpy as np

from mlprogram import Environment
from mlprogram.languages.csg import Interpreter


class TransformCanvas:
    def per_canvas(self, canvas: np.array) -> torch.Tensor:
        tensor = torch.tensor(canvas)
        return tensor.float().unsqueeze(0) - 0.5

    def __call__(self, entry: Environment) -> Environment:
        if "test_case" in entry.inputs:
            # TODO use input of test case (?)
            _, output = entry.inputs["test_case"]
            entry.states["test_case_tensor"] = self.per_canvas(output)
        if "variables" in entry.inputs:
            variables = entry.inputs["variables"]
            s = entry.states["test_case_tensor"].shape
            if len(variables) == 0:
                entry.states["variables_tensor"] = torch.zeros((0, *s))
            else:
                entry.states["variables_tensor"] = torch.stack([
                    self.per_canvas(canvas) for canvas in variables
                ])
        return entry


class AddTestCases:
    def __init__(self, interpreter: Interpreter, reference: bool = False):
        self.interpreter = interpreter
        self.reference = reference

    def __call__(self, entry: Environment) -> Environment:
        ground_truth = entry.supervisions["ground_truth"]
        if self.reference:
            results = self.interpreter.eval_references(ground_truth, None)
            entry.inputs["test_case"] = \
                (None, results[ground_truth.statements[-1].reference])
        else:
            entry.inputs["test_case"] = \
                (None, self.interpreter.eval(ground_truth, None))
        return entry
