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
            {"text_query": "foo", "ground_truth": "1 + 1"},
            set(["ground_truth"])
        )
    ]
    torch.save({"train": data, "test": data, "valid": data}, path)


def create_dummy_nl2bash(path):
    data = [
        Environment(
            {"text_query": "foo", "ground_truth": "echoa"},
            set(["ground_truth"])
        )
    ]
    torch.save({"train": data, "test": data, "valid": data}, path)


def create_dummy_deepfix(path):
    data = ([Environment({"code": "int main(){ return 0;}"})] * 100) + \
        ([Environment({"code": "int main(){ return 0}"})] * 10)
    torch.save(data, path)


if __name__ == "__main__":
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    create_dummy_django(os.path.join(DEFAULT_CACHE_DIR, "django.pt"))
    create_dummy_hearthstone(os.path.join(DEFAULT_CACHE_DIR, "hearthstone.pt"))
    create_dummy_nl2bash(os.path.join(DEFAULT_CACHE_DIR, "nl2bash.pt"))
    create_dummy_deepfix(os.path.join(DEFAULT_CACHE_DIR, "deepfix.pt"))
