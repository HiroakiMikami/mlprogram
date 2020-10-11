import json
from typing import Dict

from mlprogram.utils.data import ListDataset
from mlprogram import logging
from mlprogram import Environment

logger = logging.Logger(__name__)


def load(path: str) -> Dict[str, ListDataset]:
    with open(path) as file:
        data = json.load(file)
    dataset = {}
    dataset["train"] = ListDataset(
        [Environment.create(x) for x in data["train"]]
    )
    dataset["test"] = ListDataset(
        [Environment.create(x) for x in data["test"]]
    )
    dataset["valid"] = ListDataset(
        [Environment.create(x) for x in data["valid"]]
    )
    return dataset
