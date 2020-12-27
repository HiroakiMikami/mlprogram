import os
from typing import Any, Dict, Optional

import python_config
from pytorch_pfn_extras.config import Config

from mlprogram.entrypoint.types import types
from mlprogram.functools import file_cache


def with_file_cache(path, config, types):
    def restore(d):
        if not isinstance(d, dict):
            return d
        if "_type" in d and \
                d["_type"] == "with_file_cache":
            t = d.pop("_type")
            d["type"] = t
            config = d.pop("config")
            d = {key: restore(value) for key, value in d.items()}
            d["config"] = config
        elif "_type" in d:
            t = d.pop("_type")
            d = {key: restore(value) for key, value in d.items()}
            d["type"] = t
        return d

    @file_cache(path)
    def f():
        return Config(restore(config), types)["/"]
    return f()


def parse_config(configs: Dict[str, Any],
                 custom_types: Optional[Dict[str, Any]] = None) -> Config:
    _types = {k: v for k, v in types.items()}
    if custom_types is not None:
        for k, v in custom_types.items():
            _types[k] = v
    _types["with_file_cache"] = \
        lambda path, config: with_file_cache(path, config, _types)

    def convert(d: Any, rename: bool) -> Any:
        if not isinstance(d, dict):
            return d
        if rename:
            if "type" in d:
                t = d.pop("type")
                d = {key: convert(value, True) for key, value in d.items()}
                d["_type"] = t
            else:
                d = {key: convert(value, True) for key, value in d.items()}
        else:
            if "type" in d and d["type"] == "with_file_cache":
                config = d.pop("config")
                d = {key: convert(value, False) for key, value in d.items()}
                d["config"] = convert(config, True)
            else:
                d = {key: convert(value, False) for key, value in d.items()}
        return d

    return Config(convert(configs, False), _types)


def load_config(file: str) -> Dict[str, Any]:
    with open(file) as f:
        orig = python_config.load(f)
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
        main = python_config.load(f)
    for key, value in main.items():
        if key == "imports":
            continue
        configs[key] = value
    return configs


def parse_config_file(file: str) -> Config:
    return parse_config(load_config(file))
