import requests
from typing import Callable, Tuple, Dict

from mlprogram import logging
from mlprogram import Environment
from mlprogram.utils.data import ListDataset
from mlprogram.datasets.django.format_annotations import format_annotations

logger = logging.Logger(__name__)

BASE_PATH = "https://raw.githubusercontent.com/" + \
    "odashi/ase15-django-dataset/master/django/"


def default_get(path: str) -> str:
    return requests.get(path).text


def download(base_path: str = BASE_PATH,
             get: Callable[[str], str] = default_get,
             num_train: int = 16000, num_test: int = 1000) \
         -> Dict[str, ListDataset]:
    logger.info("Download django dataset")
    annotation = get(BASE_PATH + "all.anno").split("\n")
    annotation = format_annotations(annotation)
    code = get(BASE_PATH + "all.code").split("\n")

    def to_sample(elem: Tuple[str, str]) -> Environment:
        anno, code = elem
        return Environment(
            inputs={"input": anno},
            supervisions={"ground_truth": code}
        )
    samples = list(map(to_sample, zip(annotation, code)))

    train = ListDataset(samples[:num_train])
    test = ListDataset(samples[num_train:num_train + num_test])
    valid = ListDataset(samples[num_train + num_test:])

    return {"train": train, "test": test, "valid": valid}
