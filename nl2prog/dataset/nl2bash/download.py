import subprocess
import tempfile
import os
import json
from typing import Dict, List

from nl2prog.utils.data import Entry, ListDataset


def download(bin_dir: str = "bin") -> Dict[str, ListDataset]:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [os.path.join(bin_dir, "download_nl2bash.bash"), tmpdir])

        with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                               "train.filtered.json")) as file:
            train = json.load(file)
        with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                               "test.filtered.json")) as file:
            test = json.load(file)
        with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                               "dev.filtered.json")) as file:
            valid = json.load(file)

    def to_entry(example: Dict[str, str]) -> Entry:
        return Entry(example["source"], example["target"])

    def to_group(group) -> List[Entry]:
        return [to_entry(example) for example in group["examples"]]
    dataset = {}
    dataset["train"] = \
        ListDataset([to_group(group) for group in train["examples"]])
    dataset["test"] = \
        ListDataset([to_group(group) for group in test["examples"]])
    dataset["valid"] = \
        ListDataset([to_group(group) for group in valid["examples"]])

    return dataset
