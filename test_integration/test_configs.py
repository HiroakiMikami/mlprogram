import logging
import os
import sys
import tempfile
from typing import List, Tuple

from tools.launch import launch_multiprocess

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class TestConfig(object):
    def launch_config(self, configs: List[Tuple[str, int]]):
        with tempfile.TemporaryDirectory() as tmpdir:
            for config, n_process in configs:
                launch_multiprocess(config, "test", tmpdir, n_process)

    def test_django_nl2code(self):
        self.launch_config([
            (os.path.join("configs", "django", "nl2code_train.py"), 0),
            (os.path.join("configs", "django", "nl2code_evaluate.py"), 0),
        ])

    def test_hearthstone_nl2code(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "nl2code_train.py"), 0),
            (os.path.join("configs", "hearthstone", "nl2code_evaluate.py"),
             0),
        ])

    def test_hearthstone_treegen(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "treegen_train.py"), 0),
            (os.path.join("configs", "hearthstone", "treegen_evaluate.py"),
             0)
        ])

    def test_nl2bash_nl2code(self):
        self.launch_config([
            (os.path.join("configs", "nl2bash", "nl2code_train.py"), 0),
            (os.path.join("configs", "nl2bash", "nl2code_evaluate.py"), 0)
        ])

    def test_csg_small_pbe_without_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "without_repl_train_small.py"), 0),
            (os.path.join("configs", "csg",
                          "without_repl_evaluate_small.py"), 0)
        ])

    def test_csg_large_pbe_without_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "without_repl_train_large.py"), 0),
            (os.path.join("configs", "csg",
                          "without_repl_evaluate_large.py"), 0),
        ])

    def test_csg_small_pbe_with_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "pbe_with_repl_pretrain_small.py"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_train_small.py"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_evaluate_small.py"), 0)
        ])

    def test_csg_large_pbe_with_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "pbe_with_repl_pretrain_large.py"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_train_large.py"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_evaluate_large.py"), 0),
        ])

    def test_deepfix_baseline(self):
        self.launch_config([
            (os.path.join("configs", "deepfix",
                          "baseline_train.py"), 0),
            (os.path.join("configs", "deepfix",
                          "baseline_evaluate.py"), 0),
        ])
