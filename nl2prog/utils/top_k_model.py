import os
import torch
from .top_k_element import TopKElement


class TopKModel:
    """
    Hold top-k models
    """
    def __init__(self, k: int, directory: str):
        """
        Parameters
        ----------
        k: int
            The maximum number of models to be saved
        directory: int
            The directory that models will be saved
        """
        self._models = TopKElement(k, lambda path: os.remove(path))
        self._directory = directory

        for model in os.listdir(directory):
            path = os.path.join(directory, model)
            score = torch.load(path, map_location="cpu")["score"]
            self._models.add(score, path)

    def save(self, score: float, name: str, model: torch.nn.Module):
        """
        Save a model

        Parameters
        ----------
        score: float
        name: str
        model: torch.nn.Module
        """
        path = os.path.join(self._directory, f"model_{name}.pickle")
        result = {"score": score, "model": model.state_dict()}
        torch.save(result, path)
        self._models.add(score, path)
