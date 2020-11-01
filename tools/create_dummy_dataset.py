import os

import torch

from mlprogram import Environment, logging
from mlprogram.datasets import DEFAULT_CACHE_DIR

logger = logging.Logger(__name__)


def create_dummy_django(path):
    data = {
        "annotation": ["foo"] * 20000,
        "code": ["1 + 1"] * 20000
    }
    torch.save(data, path)


def create_dummy_hearthstone(path):
    data = [
        Environment(
            inputs={"text_query": "foo"},
            supervisions={"ground_truth": "1 + 1"}
        )
    ]
    torch.save({"train": data, "test": data, "valid": data}, path)


def create_dummy_nl2bash(path):
    data = [
        Environment(
            inputs={"text_query": "foo"},
            supervisions={"ground_truth": "echoa"}
        )
    ]
    torch.save({"train": data, "test": data, "valid": data}, path)


if __name__ == "__main__":
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    create_dummy_django(os.path.join(DEFAULT_CACHE_DIR, "django.pt"))
    create_dummy_hearthstone(os.path.join(DEFAULT_CACHE_DIR, "hearthstone.pt"))
    create_dummy_nl2bash(os.path.join(DEFAULT_CACHE_DIR, "nl2bash.pt"))
