import yaml
from viemonet.config.env import load_env
from typing import Any, Dict
import torch

from viemonet.constant import TRAINING_CONFIG_PATH
from viemonet.schemas.config import (
    Config, 
    TrainConfig, 
    FoundationModelConfig, 
    ModelHeadsConfig, 
    EKBConfig
)


def load_config() -> Config:
    load_env()
    with open(TRAINING_CONFIG_PATH, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    return Config(
        training_setting=TrainConfig(**raw_cfg['training_setting']),
        foundation_models=FoundationModelConfig(**raw_cfg['foundation_models']),
        model=ModelHeadsConfig(**raw_cfg['model']),
        emotion_knowledge_base=EKBConfig(**raw_cfg['emotion_knowledge_base']),
    )


config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")