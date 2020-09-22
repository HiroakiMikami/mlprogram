import os
from torch import nn
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extension
from mlprogram.utils import TopKModel
from mlprogram import logging

logger = logging.Logger(__name__)


class SaveTopKModel(extension.Extension):
    def __init__(self, model_dir: str, n_model: int, key: str,
                 model: nn.Module, maximize: bool = True):
        super().__init__()
        os.makedirs(model_dir, exist_ok=True)
        self.key = key
        self.maximize = maximize
        self.top_k_model = TopKModel(n_model, model_dir)
        self.model = model

    def __call__(self, manager: training.ExtensionsManager) -> None:
        if self.key in manager.observation:
            score = manager.observation[self.key]
            logger.debug("Update top-K model: score={score}")

            if not self.maximize:
                score = -score
            self.top_k_model.save(score,
                                  f"{manager.iteration}", self.model)
