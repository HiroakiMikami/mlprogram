import unittest
import tempfile
import os
import logging
import sys
from typing import List
from tools.launch import launch

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class ConfigTest(unittest.TestCase):
    def launch_config(self, configs: List[str]):
        with tempfile.TemporaryDirectory() as tmpdir:
            for config in configs:
                launch(config, "test", tmpdir)

    def test_django_nl2code(self):
        self.launch_config([
            os.path.join("configs", "django", "nl2code_train.yaml"),
            os.path.join("configs", "django", "nl2code_evaluate.yaml")
        ])

    def test_hearthstone_nl2code(self):
        self.launch_config([
            os.path.join("configs", "hearthstone", "nl2code_train.yaml"),
            os.path.join("configs", "hearthstone", "nl2code_evaluate.yaml")
        ])

    def test_hearthstone_treegen(self):
        self.launch_config([
            os.path.join("configs", "hearthstone", "treegen_train.yaml"),
            os.path.join("configs", "hearthstone", "treegen_evaluate.yaml")
        ])
    # TODO skip this dataset because nl2bash.download failed
    """
    def test_nl2bash_nl2code(self):
        self.run([
            os.path.join("configs", "nl2bash", "nl2code_train.yaml"),
            os.path.join("configs", "nl2bash", "nl2code_evaluate.yaml")
        ])
    """

    def test_csg_small_pbe_with_repl(self):
        self.launch_config([
            os.path.join("configs", "csg",
                         "pbe_with_repl_pretrain_small.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_train_small.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_evaluate_small.yaml")
        ])

    def test_csg_large_pbe_with_repl(self):
        self.launch_config([
            os.path.join("configs", "csg",
                         "pbe_with_repl_pretrain_large.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_train_large.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_evaluate_large.yaml")
        ])


if __name__ == "__main__":
    unittest.main()
