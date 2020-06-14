import requests
from typing import Callable, Dict

from mlprogram.utils.data import ListDataset

BASE_PATH = "https://raw.githubusercontent.com/" + \
    "deepmind/card2code/master/third_party/hearthstone/"


def default_get(path: str) -> str:
    return requests.get(path).text


def download(base_path: str = BASE_PATH,
             get: Callable[[str], str] = default_get) \
        -> Dict[str, ListDataset]:
    dataset = {}
    for name in ["train", "dev", "test"]:
        target = name
        if name == "test":
            target = "valid"
        if name == "dev":
            target = "test"
        query = get(f"{base_path}/{name}_hs.in").split("\n")
        code = get(f"{base_path}/{name}_hs.out").split("\n")
        code = [c.replace("§", "\n").replace("and \\", "and ") for c in code]
        groups = []
        for q, c in zip(query, code):
            if q == "" and c == "":
                continue
            groups.append({"input": [q], "ground_truth": [c]})
        dataset[target] = ListDataset(groups)
    return dataset
