import os
from torch import nn
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extension
from mlprogram.utils import TopKModel
from mlprogram.utils import logging

logger = logging.Logger(__name__)


class SaveTopKModel(extension.Extension):
    def __init__(self, model_dir: str, n_model: int, key: str,
                 model: nn.Module):
        super().__init__()
        os.makedirs(model_dir, exist_ok=True)
        self.key = key
        self.top_k_model = TopKModel(n_model, model_dir)
        self.model = model

    def __call__(self, manager: training.ExtensionsManager) -> None:
        # Find LogReport (TODO)
        score = None
        for _ext in manager._extensions.values():
            ext = _ext.extension
            if isinstance(ext, training.extensions.LogReport):
                score = ext.log[-1][self.key]
        if score is None:
            score = manager.observation[self.key]

        logger.debug("Update top-K model: score={score}")
        self.top_k_model.save(score,
                              f"{manager.iteration}", self.model)
