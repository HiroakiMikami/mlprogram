import logging
import os
import sys
import tempfile

try:
    from fairseq import optim
except:  # noqa
    optim = None
from typing import List, Tuple

from mlprogram.launch.launch import launch_multiprocess

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def launch_config(files: List[Tuple[str, int]]):
    with tempfile.TemporaryDirectory() as tmpdir:
        for config, n_process in files:
            launch_multiprocess(config, "test", tmpdir, n_process, [])


def test_deepfix_baseline():
    launch_config([
        (os.path.join("examples", "deepfix", "baseline_train.py"), 0),
        (os.path.join("examples", "deepfix", "baseline_evaluate.py"), 0),
    ])
