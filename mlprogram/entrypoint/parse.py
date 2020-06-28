import yaml
from typing import Dict, Any
from pytorch_pfn_extras.config import Config
from .types import types


def parse_config(configs: Dict[str, Any]) -> Config:
    return Config(configs, types)


def parse_config_file(file: str) -> Config:
    with open(file) as f:
        configs = yaml.load(f)
    return parse_config(configs)
