import torch
import os
from torch import nn
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extension
from mlprogram.collections import TopKElement
from mlprogram import logging

logger = logging.Logger(__name__)


class SaveTopKModel(extension.Extension):
    def __init__(self, model_dir: str, n_model: int, key: str,
                 model: nn.Module, maximize: bool = True):
        super().__init__()
        os.makedirs(model_dir, exist_ok=True)
        self.key = key
        self.maximize = maximize
        self.models = TopKElement(n_model, lambda path: os.remove(path))
        self.model_dir = model_dir
        self.model = model

        logger.info("Load saved top-k models")
        for model in os.listdir(model_dir):
            logger.debug(f"Load {model} and add to top-k model")
            path = os.path.join(model_dir, model)
            score = torch.load(path, map_location="cpu")["score"]
            self.models.add(score, path)

    def __call__(self, manager: training.ExtensionsManager) -> None:
        if self.key in manager.observation:
            score = manager.observation[self.key]
            logger.debug("Update top-K model: score={score}")

            if not self.maximize:
                score = -score
            path = os.path.join(self.model_dir,
                                f"model_{manager.iteration}.pt")
            result = {"score": score, "model": self.model.state_dict()}
            torch.save(result, path)
            self.models.add(score, path)
