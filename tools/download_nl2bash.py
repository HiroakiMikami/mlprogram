#! /bin/env python3
import tempfile
import subprocess
import os
import json
from mlprogram import logging
from mlprogram import Environment

logger = logging.Logger(__name__)


def download(path: str) -> None:
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
                                   f"{name}.nl.filtered")) as file:
                inputs = list(file.readlines())
            with open(os.path.join(tmpdir, "nl2bash", "data", "bash",
                                   f"{name}.cm.filtered")) as file:
                ground_truths = list(file.readlines())
            return [
                Environment(
                    inputs={"input": input},
                    supervisions={"ground_truth": ground_truth}
                ).to_dict()
                for input, ground_truth in zip(inputs, ground_truths)
            ]

        dataset = {}
        dataset["train"] = load("train")
        dataset["test"] = load("dev")
        dataset["valid"] = load("test")

        with open(path, "w") as file:
            json.dump(dataset, file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    download(args.path)
