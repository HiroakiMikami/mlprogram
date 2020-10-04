import torch
import numpy as np
from typing import Dict, Any, List

from mlprogram.languages.csg import Interpreter


class TransformCanvas:
    def __init__(self, targets: List[str]):
        assert np.all([target in set(["input", "variables"])
                       for target in targets])
        self.targets = set(targets)

    def per_canvas(self, canvas: np.array) -> torch.Tensor:
        tensor = torch.tensor(canvas)
        return tensor.float().unsqueeze(0) - 0.5

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if "input" in self.targets:
            # TODO use both input and output?
            entry["processed_input"] = self.per_canvas(entry["input"][1])
        if "variables" in self.targets:
            variables = entry["variables"]
            s = entry["processed_input"].shape
            if len(variables) == 0:
                entry["variables"] = torch.zeros((0, *s))
            else:
                entry["variables"] = torch.stack([
                    self.per_canvas(canvas) for canvas in variables
                ])
        return entry


class AddTestCases:
    def __init__(self, interpreter: Interpreter, reference: bool = False):
        self.interpreter = interpreter
        self.reference = reference

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        ground_truth = entry["ground_truth"]
        if self.reference:
            results = self.interpreter.eval_references(ground_truth, None)
            entry["input"] = \
                (None, results[ground_truth.statements[-1].reference])
        else:
            entry["input"] = \
                (None, self.interpreter.eval(ground_truth, None))
        return entry
