from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
import torch.nn as nn

from viemonet.config import config


class VisoBERTLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.foundation_models.visobert.model_name)
        self._apply_lora_and_freeze()

    def _apply_lora_and_freeze(self):
        lora_core_config = config.training_setting.lora
        lora_config = LoraConfig(
            r=lora_core_config.r,
            lora_alpha=lora_core_config.alpha,
            lora_dropout=lora_core_config.dropout,
            bias=lora_core_config.bias,
            target_modules=lora_core_config.target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        for name, p in self.model.named_parameters():
            p.requires_grad = "lora_" in name

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
