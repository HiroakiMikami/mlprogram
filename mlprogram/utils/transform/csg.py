import torch
import numpy as np
from typing import Dict, Any


class TransformCanvas:
    def per_canvas(self, canvas: np.array) -> torch.Tensor:
        tensor = torch.tensor(canvas)
        return tensor.float().unsqueeze(0) - 0.5

    def __call__(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if "input" in entry:
            entry["input"] = self.per_canvas(entry["input"])
        if "variables" in entry:
            entry["variables"] = {
                ref: self.per_canvas(canvas)
                for ref, canvas in entry["variables"].items()
            }
        return entry
