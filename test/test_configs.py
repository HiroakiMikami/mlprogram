import unittest
import tempfile
import gin
import os
import tools.launch  # noqa
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


class ConfigTest(unittest.TestCase):
    def nl2prog(self, train_config, evaluate_config):
        with tempfile.TemporaryDirectory() as tmpdir:
            gin.clear_config()
            gin.parse_config_file(train_config)
            # Modify configs for testing
            gin.parse_config("nl2prog.gin.nl2prog.train.num_epochs = 0.001")
            gin.parse_config("torch.device.type_str = \"cpu\"")
            gin.parse_config(
                "nl2prog.gin.nl2prog.train.workspace_dir = "
                f"\"{tmpdir}/workspace\"")
            gin.parse_config(
                f"nl2prog.gin.nl2prog.train.output_dir = \"{tmpdir}/output\"")
            # Train
            tools.launch.entrypoint()

            gin.clear_config()
            gin.parse_config_file(evaluate_config)
            # Modify configs for testing
            gin.parse_config("torch.device.type_str = \"cpu\"")
            gin.parse_config("nl2prog.gin.nl2prog.evaluate.n_samples = 1")
            gin.parse_config(
                "nl2prog.gin.nl2prog.evaluate.workspace_dir = "
                f"\"{tmpdir}/workspace\"")
            gin.parse_config(
                "nl2prog.gin.nl2prog.evaluate.input_dir = "
                f"\"{tmpdir}/output\"")
            gin.parse_config(
                "nl2prog.gin.nl2prog.evaluate.output_dir = "
                f"\"{tmpdir}/output\"")
            gin.parse_config(
                "nl2prog.utils.CommonBeamSearchSynthesizer.create.max_steps = "
                "2")
            # Evaluate
            tools.launch.entrypoint()

    def test_django_nl2code(self):
        self.nl2prog(
            os.path.join("configs", "django", "nl2code_train.gin"),
            os.path.join("configs", "django", "nl2code_evaluate.gin"))

    def test_hearthstone_nl2code(self):
        self.nl2prog(
            os.path.join("configs", "hearthstone", "nl2code_train.gin"),
            os.path.join("configs", "hearthstone", "nl2code_evaluate.gin"))

    def test_hearthstone_treegen(self):
        self.nl2prog(
            os.path.join("configs", "hearthstone", "treegen_train.gin"),
            os.path.join("configs", "hearthstone", "treegen_evaluate.gin"))

    # TODO skip this dataset because nl2bash.download failed
    """
    def test_nl2bash_nl2code(self):
        self.nl2prog(
            os.path.join("configs", "nl2bash", "nl2code_train.gin"),
            os.path.join("configs", "nl2bash", "nl2code_evaluate.gin"))
    """


if __name__ == "__main__":
    unittest.main()
