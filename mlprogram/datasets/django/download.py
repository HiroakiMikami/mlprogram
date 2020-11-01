import os
from typing import Callable, Dict, Tuple

import requests

from mlprogram import Environment, logging
from mlprogram.datasets import DEFAULT_CACHE_DIR
from mlprogram.datasets.django.format_annotations import format_annotations
from mlprogram.functools import file_cache
from mlprogram.utils.data import ListDataset

logger = logging.Logger(__name__)

BASE_PATH = "https://raw.githubusercontent.com/" + \
    "odashi/ase15-django-dataset/master/django/"


def default_get(path: str) -> str:
    return requests.get(path).text


def download(cache_path: str = os.path.join(DEFAULT_CACHE_DIR, "django.pt"),
             base_path: str = BASE_PATH,
             get: Callable[[str], str] = default_get,
             num_train: int = 16000, num_test: int = 1000) \
        -> Dict[str, ListDataset]:

    @file_cache(cache_path)
    def _download():
        return {
            "annotation": format_annotations(
                get(BASE_PATH + "all.anno").split("\n")),
            "code": get(BASE_PATH + "all.code").split("\n")
        }
    data = _download()
    annotation = data["annotation"]
    code = data["code"]

    def to_sample(elem: Tuple[str, str]) -> Environment:
        anno, code = elem
        return Environment(
            inputs={"text_query": anno},
            supervisions={"ground_truth": code}
        )
    samples = list(map(to_sample, zip(annotation, code)))

    data = {
        "train": samples[:num_train],
        "test": samples[num_train:num_train + num_test],
        "valid": samples[num_train + num_test:]
    }

    return {key: ListDataset(value) for key, value in data.items()}
