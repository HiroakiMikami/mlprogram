import torch
import numpy as np
from typing import Dict, Any, List


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
            entry["processed_input"] = self.per_canvas(entry["input"])
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
