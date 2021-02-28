import logging
import os
import sys
import tempfile

import pytest

try:
    from fairseq import optim
except:  # noqa
    optim = None
from typing import List, Tuple

from tools.launch import launch_multiprocess

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestConfig(object):
    def launch_config(self, configs: List[Tuple[str, int]]):
        with tempfile.TemporaryDirectory() as tmpdir:
            for config, n_process in configs:
                launch_multiprocess(config, "test", tmpdir, n_process)

    def test_django_baseline(self):
        self.launch_config([
            (os.path.join("configs", "django", "baseline_train.py"), 0),
            (os.path.join("configs", "django", "baseline_evaluate.py"), 0),
        ])

    def test_hearthstone_baseline(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "baseline_train.py"), 0),
            (os.path.join("configs", "hearthstone", "baseline_evaluate.py"),
             0),
        ])

    def test_hearthstone_nl2code(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "nl2code_train.py"), 0),
            (os.path.join("configs", "hearthstone", "nl2code_evaluate.py"),
             0),
        ])

    @pytest.mark.skipif(optim is None, reason="fairseq is not installed")
    def test_hearthstone_treegen(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "treegen_train.py"), 0),
            (os.path.join("configs", "hearthstone", "treegen_evaluate.py"),
             0)
        ])

    """
    def test_nl2bash_baseline(self):
        self.launch_config([
            (os.path.join("configs", "nl2bash", "baseline_train.py"), 0),
            (os.path.join("configs", "nl2bash", "baseline_evaluate.py"), 0)
        ])
    """

    def test_csg_pbe_baseline(self):
        self.launch_config([
            (os.path.join("configs", "csg", "baseline_train.py"), 0),
            (os.path.join("configs", "csg", "baseline_evaluate_short.py"), 0),
            (os.path.join("configs", "csg", "baseline_evaluate_long.py"), 0),
            (
                os.path.join(
                    "configs", "csg", "baseline_evaluate_rl_synthesizer_short.py"
                ),
                0
            ),
            (
                os.path.join(
                    "configs", "csg", "baseline_evaluate_rl_synthesizer_long.py"
                ),
                0
            ),
        ])

    def test_csg_pbe_with_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg", "pbe_with_repl_pretrain.py"), 0),
            (os.path.join("configs", "csg", "pbe_with_repl_train.py"), 0),
            (os.path.join("configs", "csg", "pbe_with_repl_evaluate_short.py"), 0),
            (os.path.join("configs", "csg", "pbe_with_repl_evaluate_long.py"), 0),
        ])

    def test_deepfix_baseline(self):
        self.launch_config([
            (os.path.join("configs", "deepfix", "baseline_train.py"), 0),
            (os.path.join("configs", "deepfix", "baseline_evaluate.py"), 0),
        ])
