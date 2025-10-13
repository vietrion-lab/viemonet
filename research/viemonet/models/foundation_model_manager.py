from transformers import AutoTokenizer

from viemonet.constant import FOUNDATION_MODEL_LIST
from research.viemonet.models.foundation_models.phobert_lora import PhoBERTLoRA
from research.viemonet.models.foundation_models.vit5_lora import ViT5LoRA
from viemonet.config import config


class FoundationModelManager:
    def __init__(self):
        pass

    def get_model(self, model_name):
        if model_name == 'phobert':
            model = PhoBERTLoRA()
            tokenizer = AutoTokenizer.from_pretrained(config.foundation_models.phobert.model_name)
            return model, tokenizer, config.foundation_models.phobert.embedding_dim
        # elif model_name == 'vit5':
        #     model = ViT5LoRA()
        #     tokenizer = model.tokenizer
        #     return model, tokenizer, config.foundation_models.vit5.embedding_dim
        else:
            raise ValueError(f"Unknown foundation model: {model_name}. Must be one of {FOUNDATION_MODEL_LIST}")