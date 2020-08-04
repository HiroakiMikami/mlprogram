import os
import yaml
from typing import Dict, Any
from pytorch_pfn_extras.config import Config
from mlprogram.entrypoint.types import types


def parse_config(configs: Dict[str, Any]) -> Config:
    return Config(configs, types)


def load_config(file: str) -> Dict[str, Any]:
    with open(file) as f:
        orig = yaml.load(f)
    dir = os.path.dirname(file)
    if "imports" in orig:
        imports = orig.pop("imports")
    else:
        imports = []
    configs = {}
    for subfile in imports:
        subconfig = load_config(os.path.join(dir, subfile))
        for key, value in subconfig.items():
            configs[key] = value
    with open(file) as f:
        main = yaml.load(f)
    for key, value in main.items():
        if key == "imports":
            continue
        configs[key] = value
    return configs


def parse_config_file(file: str) -> Config:
    return parse_config(load_config(file))
