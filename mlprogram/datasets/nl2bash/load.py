import json
from typing import Dict

from mlprogram.utils.data import ListDataset
from mlprogram.utils import logging

logger = logging.Logger(__name__)


def load(path: str) -> Dict[str, ListDataset]:
    with open(path) as file:
        data = json.load(file)
    dataset = {}
    dataset["train"] = ListDataset(data["train"])
    dataset["test"] = ListDataset(data["test"])
    dataset["valid"] = ListDataset(data["valid"])
    return dataset
