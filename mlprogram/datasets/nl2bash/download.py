import os
import subprocess
import tempfile
from typing import Dict

from mlprogram import logging
from mlprogram.builtins import Environment
from mlprogram.datasets import DEFAULT_CACHE_DIR
from mlprogram.functools import file_cache
from mlprogram.utils.data import ListDataset

logger = logging.Logger(__name__)


def download(cache_path: str = os.path.join(DEFAULT_CACHE_DIR, "nl2bash.pt")
             ) -> Dict[str, ListDataset]:

    @file_cache(cache_path)
    def _download():
        logger.info("Download nl2bash dataset")
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "download.bash"), "w") as file:
                file.write("""
#! /bin/env bash
tmpdir=$1

python -m venv $tmpdir/env
source $tmpdir/env/bin/activate

git clone --depth 1 https://github.com/TellinaTool/nl2bash $tmpdir/nl2bash
pip install tensorflow
pip install -r $tmpdir/nl2bash/requirements.txt
make -C $tmpdir/nl2bash
set +u
export PYTHONPATH=$tmpdir/nl2bash:$PYTHONPATH
set -u
make -C $tmpdir/nl2bash/scripts data
""")
            subprocess.run(
                ["bash", os.path.join(tmpdir, "download.bash"), tmpdir])

            def load(name: str):
                with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                                       f"{name}.nl.filtered"),
                          encoding="utf-8") as file:
                    inputs = list(file.readlines())
                with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                                       f"{name}.cm.filtered"),
                          encoding="utf-8") as file:
                    ground_truths = list(file.readlines())
                return [
                    Environment(
                        {
                            "text_query": input,
                            "ground_truth": ground_truth
                        },
                        set(["ground_truth"])
                    )
                    for input, ground_truth in zip(inputs, ground_truths)
                ]

            dataset = {}
            dataset["train"] = load("train")
            dataset["test"] = load("dev")
            dataset["valid"] = load("test")
        return dataset

    dataset = _download()
    return {key: ListDataset(value) for key, value in dataset.items()}
