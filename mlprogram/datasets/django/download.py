import requests
import logging
from typing import Callable, Tuple, List, Dict, Any

from mlprogram.utils.data import ListDataset
from .format_annotations import format_annotations

logger = logging.getLogger(__name__)

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

    def to_group(elem: Tuple[str, str]) -> Dict[str, List[Any]]:
        anno, code = elem
        return {"input": [anno], "ground_truth": [code]}
    data = list(map(to_group, zip(annotation, code)))

    train = ListDataset(data[:num_train])
    test = ListDataset(data[num_train:num_train + num_test])
    valid = ListDataset(data[num_train + num_test:])

    return {"train": train, "test": test, "valid": valid}
