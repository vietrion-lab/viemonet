from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
import torch.nn as nn

from viemonet.config import config


class ViT5LoRA(nn.Module):
    """LoRA-wrapped ViT5 encoder with backward-compatible checkpoint loading.
    
    ViT5 is a T5-based encoder-decoder model. For sentiment classification,
    we primarily use the encoder part and apply LoRA to make it parameter-efficient.
    
    Similar to PhoBERTLoRA, this wrapper handles:
    - Token embedding resizing for emoji tokens
    - LoRA injection for efficient fine-tuning
    - State dict compatibility for checkpoint loading
    """
        
    def __init__(self, tokenizer=None, projected_emoji_vectors=None):
        """Initialize ViT5 with optional tokenizer extension and emoji embeddings.

        Args:
            tokenizer: Pre-extended tokenizer with emoji tokens
            projected_emoji_vectors: dict[str, torch.Tensor] for initializing emoji embeddings
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(config.foundation_model.model_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.foundation_model.tokenizer_name)
        
        # Resize token embeddings if tokenizer was extended
        base_embed = self.model.get_input_embeddings()
        if base_embed.num_embeddings < len(self.tokenizer):
            old_num = base_embed.num_embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Initialize new emoji token embeddings with projected vectors
            if projected_emoji_vectors:
                import torch
                with torch.no_grad():
                    embed_weight = self.model.get_input_embeddings().weight
                    vocab = self.tokenizer.get_vocab()
                    for emo, vec in projected_emoji_vectors.items():
                        if emo in vocab:
                            idx = vocab[emo]
                            if idx >= old_num and vec.shape[-1] == embed_weight.size(1):
                                embed_weight[idx] = vec.to(embed_weight.device)
        
        self._apply_lora_and_freeze()

    def _apply_lora_and_freeze(self):
        """Apply LoRA to encoder layers and freeze base model parameters."""
        lora_core_config = config.training.lora
        lora_config = LoraConfig(
            r=lora_core_config.r,
            lora_alpha=lora_core_config.alpha,
            lora_dropout=lora_core_config.dropout,
            bias=lora_core_config.bias,
            # For T5/ViT5: target query and value projection in attention layers
            target_modules=lora_core_config.target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Only LoRA parameters are trainable
        for name, p in self.model.named_parameters():
            p.requires_grad = "lora_" in name

    def forward(self, input_ids, attention_mask):  # type: ignore[override]
        """Forward pass through ViT5 encoder.
        
        For T5 models, we use encoder_outputs to get the hidden states.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # T5 models return encoder outputs when only input_ids provided
        # We return the full output for compatibility with classification heads
        return outputs

    # ----------------------------
    # State dict compatibility API
    # ----------------------------
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        """Return inner PEFT model weights without wrapper 'model.' prefix.
        
        This ensures checkpoint compatibility across different wrapper versions.
        """
        inner = self.model.state_dict(*args, **kwargs)
        return {k: v for k, v in inner.items()}

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load state dict accepting both flattened and 'model.'-prefixed keys."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                remapped[k[len('model.'):]] = v
            else:
                remapped[k] = v
        missing, unexpected = self.model.load_state_dict(remapped, strict=strict)
        return missing, unexpected

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Handle recursive state dict loading with prefix remapping."""
        legacy_extra = prefix + 'model.'
        expected_prefix = prefix + 'model.'
        
        # Ensure both naming variants exist for compatibility
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not k.startswith(expected_prefix):
                tail = k[len(prefix):]
                candidate = expected_prefix + tail
                if candidate not in state_dict:
                    state_dict[candidate] = state_dict[k]
        
        # Delegate to underlying PEFT model
        self.model._load_from_state_dict(
            state_dict,
            expected_prefix,
            local_metadata.get('model', {}),
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
