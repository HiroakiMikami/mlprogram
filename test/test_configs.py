import unittest
import tempfile
import os
import logging
import yaml
import sys
from mlprogram.entrypoint.parse import parse_config

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)


class ConfigTest(unittest.TestCase):
    def nl2prog(self, train_config, evaluate_config):
        logger.info(f"Train: {train_config}")
        logger.info(f"Evaluate: {evaluate_config}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(train_config) as file:
                configs = yaml.load(file)
            # Modify configs for testing
            configs["main"]["length"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 3
            }
            configs["main"]["interval"] = {
                "type": "mlprogram.entrypoint.train.Iteration",
                "n": 3
            }
            configs["device"]["type_str"] = "cpu"
            configs["main"]["workspace_dir"] = f"{tmpdir}/workspace"
            configs["output_dir"] = f"{tmpdir}/output"
            # Train
            parse_config(configs)["/main"]

            with open(evaluate_config) as file:
                configs = yaml.load(file)
            # Modify configs for testing
            configs["main"]["n_samples"] = 1
            configs["device"]["type_str"] = "cpu"
            configs["main"]["workspace_dir"] = f"{tmpdir}/workspace"
            configs["output_dir"] = f"{tmpdir}/output"
            configs["synthesizer"]["max_step_size"] = 2
            # Evaluate
            parse_config(configs)["/main"]

    def test_django_nl2code(self):
        self.nl2prog(
            os.path.join("configs", "django", "nl2code_train.yaml"),
            os.path.join("configs", "django", "nl2code_evaluate.yaml"))

    def test_hearthstone_nl2code(self):
        self.nl2prog(
            os.path.join("configs", "hearthstone", "nl2code_train.yaml"),
            os.path.join("configs", "hearthstone", "nl2code_evaluate.yaml"))

    def test_hearthstone_treegen(self):
        self.nl2prog(
            os.path.join("configs", "hearthstone", "treegen_train.yaml"),
            os.path.join("configs", "hearthstone", "treegen_evaluate.yaml"))

    # TODO skip this dataset because nl2bash.download failed
    """
    def test_nl2bash_nl2code(self):
        self.mlprogram(
            os.path.join("configs", "nl2bash", "nl2code_train.yaml"),
            os.path.join("configs", "nl2bash", "nl2code_evaluate.yaml"))
    """


if __name__ == "__main__":
    unittest.main()
