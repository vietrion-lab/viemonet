import yaml
from datasets import load_dataset as lib_load_dataset

from transform_data.schemas import TransformConfig


def load_config(path) -> TransformConfig:
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return TransformConfig(**raw_cfg)

def load_dataset(path):
    return lib_load_dataset(path)