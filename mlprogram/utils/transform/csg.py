import torch
import numpy as np
from typing import Dict, Any


class TransformCanvas:
    def per_canvas(self, canvas: np.array) -> torch.Tensor:
        tensor = torch.tensor(canvas)
        return tensor.float().unsqueeze(0) - 0.5

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        entry["input"] = self.per_canvas(entry["input"])
        if "variables" in entry:
            variables = entry["variables"]
            s = entry["input"].shape
            if len(variables) == 0:
                entry["variables"] = torch.zeros((0, *s))
            else:
                entry["variables"] = torch.stack([
                    self.per_canvas(canvas) for canvas in variables
                ])
        return entry
