import yaml
from emotiment.config.env import load_env
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional

from emotiment.constant import TRAINING_CONFIG_PATH


class EarlyStoppingConfig(BaseModel):
    metric: str
    patience: int


class TrainConfig(BaseModel):
    output_dir: str
    max_length: int
    batch_train: int
    batch_eval: int
    lora_lr: float
    head_lr: float
    weight_decay: float
    warmup_ratio: float
    epochs: int
    warmup_steps: int
    param_groups: List[str]
    fp16: bool
    max_grad_norm: float
    early_stopping: EarlyStoppingConfig
    
    # LoRA section
    class LoRAConfig(BaseModel):
        r: int
        alpha: int
        dropout: float
        bias: str
        target_modules: List[str]

    lora: LoRAConfig


class ModelConfig(BaseModel):
    # Allow attributes starting with 'model_' without protected namespace warnings
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    tokenizer_name: str


class DatasetConfig(BaseModel):
    author: str
    tweet_data_name: str
    emoji_data_name: str


class LSTMHeadConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    out_dim: int


class GRUHeadConfig(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    out_dim: int

class BiGRUHeadConfig(BaseModel):
    hidden_size: int  # per direction
    num_layers: int
    dropout: float
    out_dim: int

class BiLSTMHeadConfig(BaseModel):
    hidden_size: int  # per direction
    num_layers: int
    dropout: float
    out_dim: int

class CNNHeadConfig(BaseModel):
    num_filters: int
    kernel_sizes: List[int]
    dropout: float
    out_dim: int

class LogisticRegressionHeadConfig(BaseModel):
    out_dim: int
    dropout: float

class XGBoostHeadConfig(BaseModel):
    max_depth: int
    n_estimators: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    # out_dim implied from dataset classes


class LossConfig(BaseModel):
    label_smoothing: float


class ModelHeadsConfig(BaseModel):
    lstm: LSTMHeadConfig
    gru: GRUHeadConfig
    bigru: BiGRUHeadConfig
    bilstm: BiLSTMHeadConfig
    cnn: CNNHeadConfig
    logreg: LogisticRegressionHeadConfig
    xgboost: XGBoostHeadConfig
    loss: LossConfig


class Config(BaseModel):
    training: TrainConfig
    foundation_model: ModelConfig
    dataset: DatasetConfig
    model: ModelHeadsConfig


def load_config() -> Config:
    # Load environment variables (HF tokens etc.) before accessing config-dependent resources
    load_env()
    with open(TRAINING_CONFIG_PATH, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    return Config(
        training=TrainConfig(**raw_cfg["training"]),
        foundation_model=ModelConfig(**raw_cfg["foundation_model"]),
    dataset=DatasetConfig(**raw_cfg["dataset"]),
    model=ModelHeadsConfig(**raw_cfg["model"]),
    )


config = load_config()