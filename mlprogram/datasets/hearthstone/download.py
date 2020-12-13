import os
from typing import Callable, Dict

import requests

from mlprogram import logging
from mlprogram.builtins import Environment
from mlprogram.datasets import DEFAULT_CACHE_DIR
from mlprogram.functools import file_cache
from mlprogram.utils.data import ListDataset

logger = logging.Logger(__name__)

BASE_PATH = "https://raw.githubusercontent.com/" + \
    "deepmind/card2code/master/third_party/hearthstone/"


def default_get(path: str) -> str:
    return requests.get(path).text


def download(cache_path: str = os.path.join(DEFAULT_CACHE_DIR,
                                            "hearthstone.pt"),
             base_path: str = BASE_PATH,
             get: Callable[[str], str] = default_get) \
        -> Dict[str, ListDataset]:

    @file_cache(cache_path)
    def _download():
        logger.info("Download hearthstone dataset")
        dataset = {}
        for name in ["train", "dev", "test"]:
            target = name
            if name == "test":
                target = "valid"
            if name == "dev":
                target = "test"
            query = get(f"{base_path}/{name}_hs.in").split("\n")
            code = get(f"{base_path}/{name}_hs.out").split("\n")
            code = [c.replace("§", "\n").replace("and \\", "and ")
                    for c in code]
            samples = []
            for q, c in zip(query, code):
                if q == "" and c == "":
                    continue
                samples.append(Environment(
                    {
                        "text_query": q,
                        "ground_truth": c
                    },
                    set(["ground_truth"])
                ))
            dataset[target] = samples
        return dataset
    dataset = _download()
    return {key: ListDataset(value) for key, value in dataset.items()}
