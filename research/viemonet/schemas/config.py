from pydantic import BaseModel, ConfigDict
from typing import List


class EarlyStoppingConfig(BaseModel):
    metric: str
    patience: int

# Multi-task learning section
class MultiTaskLearningConfig(BaseModel):
    alpha: float
    beta: float

# LoRA section
class LoRAConfig(BaseModel):
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: List[str]


class TrainConfig(BaseModel):
    output_dir: str
    output_structure: str
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
    best_model_metric: str      
    multi_task_learning: MultiTaskLearningConfig
    lora: LoRAConfig

class FoundationModelChildConfig(BaseModel):
    model_name: str
    tokenizer_name: str
    embedding_dim: int
    type: str

class FoundationModelConfig(BaseModel):
    phobert: FoundationModelChildConfig
    visobert: FoundationModelChildConfig
    vit5: FoundationModelChildConfig


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
    
class MetaClassifierConfig(BaseModel):
    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float

class ModelHeadsConfig(BaseModel):
    meta_cls: MetaClassifierConfig
    lstm: LSTMHeadConfig
    gru: GRUHeadConfig
    bigru: BiGRUHeadConfig
    bilstm: BiLSTMHeadConfig
    cnn: CNNHeadConfig
    logreg: LogisticRegressionHeadConfig
    xgboost: XGBoostHeadConfig
    loss: LossConfig
    
class EKBConfig(BaseModel):
    emoticon_sentiment: str
    emoji_sentiment: str

class Config(BaseModel):
    training_setting: TrainConfig
    foundation_models: FoundationModelConfig
    model: ModelHeadsConfig
    emotion_knowledge_base: EKBConfig