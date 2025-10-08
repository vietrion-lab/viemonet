import yaml
from datasets import load_dataset as lib_load_dataset

from config_schema import ConfigSchema


def load_config(path) -> ConfigSchema:
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return ConfigSchema(**raw_cfg)

def load_dataset(path):
    return lib_load_dataset(path)