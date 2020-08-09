import unittest
import tempfile
import os
import logging
import sys
from mlprogram.entrypoint.parse import parse_config, load_config

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


class ConfigTest(unittest.TestCase):
    def rl(self, pretrain_config, train_config, evaluate_config):
        logger.info(f"Pretrain: {pretrain_config}")
        logger.info(f"Train: {train_config}")
        logger.info(f"Evaluate: {evaluate_config}")
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = load_config(pretrain_config)
            # Modify configs for testing
            configs["main"]["length"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 2
            }
            configs["main"]["interval"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 2
            }
            configs["device"]["type_str"] = "cpu"
            configs["main"]["workspace_dir"] = f"{tmpdir}/workspace.pretrain"
            configs["output_dir"] = f"{tmpdir}/output"
            # Pretrain
            parse_config(configs)["/main"]

            configs = load_config(train_config)
            # Modify configs for testing
            configs["main"]["length"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 2
            }
            configs["main"]["interval"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 2
            }
            configs["device"]["type_str"] = "cpu"
            configs["main"]["workspace_dir"] = f"{tmpdir}/workspace.train"
            configs["output_dir"] = f"{tmpdir}/output"
            configs["base_synthesizer"]["max_step_size"] = 2
            configs["base_synthesizer"]["initial_particle_size"] = 1
            # Train
            parse_config(configs)["/main"]

            configs = load_config(evaluate_config)
            # Modify configs for testing
            configs["main"]["n_samples"] = 1
            configs["device"]["type_str"] = "cpu"
            configs["main"]["workspace_dir"] = f"{tmpdir}/workspace"
            configs["output_dir"] = f"{tmpdir}/output"
            configs["base_synthesizer"]["max_step_size"] = 2
            configs["base_synthesizer"]["initial_particle_size"] = 1
            # Evaluate
            parse_config(configs)["/main"]

    def test_csg_small_pbe_with_repl(self):
        self.rl(
            os.path.join("configs", "csg",
                         "pbe_with_repl_pretrain_small.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_train_small.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_evaluate_small.yaml"))

    def test_csg_large_pbe_with_repl(self):
        self.rl(
            os.path.join("configs", "csg",
                         "pbe_with_repl_pretrain_large.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_train_large.yaml"),
            os.path.join("configs", "csg",
                         "pbe_with_repl_evaluate_large.yaml"))


if __name__ == "__main__":
    unittest.main()
