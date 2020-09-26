import unittest
import os
import tempfile
from mlprogram.entrypoint.configs import load_config


class TestLoadConfig(unittest.TestCase):
    def test_simple_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "x.yaml"), "w") as file:
                file.writelines([
                    "x:",
                    "  10"
                ])
            self.assertEqual(
                {"x": 10},
                load_config(os.path.join(tmpdir, "x.yaml"))
            )

    def test_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "sub.yaml"), "w") as file:
                file.writelines([
                    "y:\n",
                    "  0\n"
                ])
            with open(os.path.join(tmpdir, "x.yaml"), "w") as file:
                file.writelines([
                    "imports:\n",
                    "  - sub.yaml\n",
                    "x: 10\n"
                ])
            self.assertEqual(
                {"x": 10, "y": 0},
                load_config(os.path.join(tmpdir, "x.yaml"))
            )

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "sub.yaml"), "w") as file:
                file.writelines([
                    "x:\n",
                    "  0\n"
                ])
            with open(os.path.join(tmpdir, "x.yaml"), "w") as file:
                file.writelines([
                    "imports:\n",
                    "  - sub.yaml\n",
                    "x: 10\n"
                ])
            self.assertEqual(
                {"x": 10},
                load_config(os.path.join(tmpdir, "x.yaml"))
            )


if __name__ == "__main__":
    unittest.main()
