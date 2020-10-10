import tempfile
import os
import logging
import sys
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
            (os.path.join("configs", "django", "nl2code_train.yaml"), 0),
            (os.path.join("configs", "django", "nl2code_evaluate.yaml"), 0),
        ])

    def test_hearthstone_nl2code(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "nl2code_train.yaml"), 0),
            (os.path.join("configs", "hearthstone", "nl2code_evaluate.yaml"),
             0),
        ])

    def test_hearthstone_treegen(self):
        self.launch_config([
            (os.path.join("configs", "hearthstone", "treegen_train.yaml"), 0),
            (os.path.join("configs", "hearthstone", "treegen_evaluate.yaml"),
             0)
        ])

    def test_nl2bash_nl2code(self):
        if not os.path.exists(os.path.join("datasets", "nl2bash.json")):
            logger.warning("nl2bash dataset is not downloaded. Skip this test")
            self.skipTest("nl2bash dataset is not downloaded. Skip this test")
            return
        self.launch_config([
            (os.path.join("configs", "nl2bash", "nl2code_train.yaml"), 0),
            (os.path.join("configs", "nl2bash", "nl2code_evaluate.yaml"), 0)
        ])

    def test_csg_small_pbe_without_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "without_repl_train_small.yaml"), 0),
            (os.path.join("configs", "csg",
                          "without_repl_evaluate_small.yaml"), 0)
        ])

    def test_csg_large_pbe_without_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "without_repl_train_large.yaml"), 0),
            (os.path.join("configs", "csg",
                          "without_repl_evaluate_large.yaml"), 0),
        ])

    def test_csg_small_pbe_with_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "pbe_with_repl_pretrain_small.yaml"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_train_small.yaml"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_evaluate_small.yaml"), 0)
        ])

    def test_csg_large_pbe_with_repl(self):
        self.launch_config([
            (os.path.join("configs", "csg",
                          "pbe_with_repl_pretrain_large.yaml"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_train_large.yaml"), 0),
            (os.path.join("configs", "csg",
                          "pbe_with_repl_evaluate_large.yaml"), 0),
        ])
