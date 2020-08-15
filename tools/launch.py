import argparse

import random
import os
import tempfile
import logging
import cProfile
import torch
from torch.autograd.profiler import profile
from typing import Optional, Any
from mlprogram.entrypoint.parse import parse_config, load_config

logger = logging.getLogger(__name__)


def modify_config_for_test(configs: Any, tmpdir: str) -> Any:
    if isinstance(configs, dict):
        out = {}
        for key, value in configs.items():
            if key == "device":
                value["type_str"] = "cpu"
            elif key == "output_dir":
                value = f"{tmpdir}/output"
            elif isinstance(value, dict) and "type" in value:
                if value["type"] in set([
                    "mlprogram.entrypoint.train_supervised",
                        "mlprogram.entrypoint.train_REINFORCE"]):
                    value["length"] = {
                        "type": "mlprogram.entrypoint.train.Iteration",
                        "n": 2
                    }
                    value["interval"] = {
                        "type": "mlprogram.entrypoint.train.Iteration",
                        "n": 2
                    }
                    value["workspace_dir"] = \
                        f"{tmpdir}/workspace.{random.randint(0, 100)}"
                elif value["type"] in set(["mlprogram.entrypoint.evaluate"]):
                    value["n_samples"] = 1
                    value["workspace_dir"] = \
                        f"{tmpdir}/workspace.{random.randint(0, 100)}"
                elif value["type"] in set(["mlprogram.synthesizers.BeamSearch",
                                           "mlprogram.synthesizers.SMC"]):
                    value["max_step_size"] = 2
                    if value["type"] == "mlprogram.synthesizers.SMC":
                        value["initial_particle_size"] = 1
                value = {k: modify_config_for_test(v, tmpdir)
                         for k, v in value.items()}
            elif isinstance(value, dict):
                value = {k: modify_config_for_test(v, tmpdir)
                         for k, v in value.items()}
            elif isinstance(value, list):
                value = [modify_config_for_test(x, tmpdir) for x in value]
            else:
                value = modify_config_for_test(value, tmpdir)
            out[key] = value
        return out
    else:
        return configs


def modify_config_for_profile(configs: Any, tmpdir: str) -> Any:
    if isinstance(configs, dict):
        out = {}
        for key, value in configs.items():
            if isinstance(value, dict) and "type" in value:
                if value["type"] in set([
                    "mlprogram.entrypoint.train_supervised",
                        "mlprogram.entrypoint.train_REINFORCE"]):
                    value["length"] = {
                        "type": "mlprogram.entrypoint.train.Iteration",
                        "n": 2
                    }
                    value["interval"] = {
                        "type": "mlprogram.entrypoint.train.Iteration",
                        "n": 2
                    }
                    value["workspace_dir"] = \
                        f"{tmpdir}/workspace.{random.randint(0, 100)}"
                    value["output_dir"] = f"{tmpdir}/output"
                elif value["type"] in set(["mlprogram.entrypoint.evaluate"]):
                    value["n_samples"] = 1
                    value["workspace_dir"] = \
                        f"{tmpdir}/workspace.{random.randint(0, 100)}"
                    value["output_dir"] = f"{tmpdir}/output"
                value = {k: modify_config_for_test(v, tmpdir)
                         for k, v in value.items()}
            elif isinstance(value, dict):
                value = {k: modify_config_for_test(v, tmpdir)
                         for k, v in value.items()}
            elif isinstance(value, list):
                value = [modify_config_for_test(x, tmpdir) for x in value]
            else:
                value = modify_config_for_test(value, tmpdir)
            out[key] = value
        return out
    else:
        return configs


def launch(config_file: str, option: Optional[str], tmpdir: str):
    logger.info(f"Launch config file: {config_file}")
    configs = load_config(config_file)
    if option == "test":
        logger.info("Modify configs for testing")
        configs = modify_config_for_test(configs, tmpdir)
    elif option == "profile":
        logger.info("Modify configs for profiling")
        output_dir = configs["output_dir"]  # TODO
        configs = modify_config_for_profile(configs, tmpdir)

    if option == "profile":
        cprofile = cProfile.Profile()
        cprofile.enable()
        with profile(use_cuda=True) as torch_prof:
            parse_config(configs)["/main"]
        cprofile.disable()
        torch.save(torch_prof, os.path.join(output_dir, "torch_profiler.pt"))
        cprofile.dump_stats(os.path.join(output_dir, "cprofile.pt"))
    else:
        parse_config(configs)["/main"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", required=True, type=str)
    parser.add_argument("--option", choices=["test", "profile"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        launch(args.config_file, args.option, tmpdir)


if __name__ == "__main__":
    main()
