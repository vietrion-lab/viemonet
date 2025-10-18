from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
import torch.nn as nn

from viemonet.config import config


class ViT5LoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(config.foundation_models.vit5.model_name)
        self._apply_lora_and_freeze()

    def _apply_lora_and_freeze(self):
        lora_core_config = config.training_setting.lora
        # T5/ViT5 uses different module names: q, k, v, o instead of query, key, value, output.dense
        # Apply LoRA only to encoder blocks, skip decoder
        lora_config = LoraConfig(
            r=lora_core_config.r,
            lora_alpha=lora_core_config.alpha,
            lora_dropout=lora_core_config.dropout,
            bias=lora_core_config.bias,
            target_modules=["q", "k", "v"],  # T5-specific attention modules
            modules_to_save=None,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Freeze all parameters first
        for name, p in self.model.named_parameters():
            p.requires_grad = False
        
        # Only enable gradients for LoRA parameters in encoder
        for name, p in self.model.named_parameters():
            if "lora_" in name and "encoder" in name:
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # For T5 models, only use the encoder for classification tasks
        # This avoids needing decoder_input_ids
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Return in the same format as BERT models for compatibility
        # Create a simple namespace object that has last_hidden_state attribute
        class EncoderOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        
        return EncoderOutput(encoder_outputs.last_hidden_state)
