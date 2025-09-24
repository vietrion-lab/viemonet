from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
import torch.nn as nn

from emotiment.config import config


class PhoBERTLoRA(nn.Module):
    """LoRA-wrapped PhoBERT encoder with backward-compatible checkpoint loading.

    Key mismatch issue observed:
        - Current model parameter names (because this wrapper holds `self.model`)
            produce keys like:  classifier_encoder.model.base_model.model.encoder.layer.0.*
        - Older checkpoints on disk lack the extra 'model.' segment right after
            'classifier_encoder.' i.e.: classifier_encoder.base_model.model.encoder.layer.0.*
        - When resuming, Torch reports a huge list of missing / unexpected keys.

    Strategy:
        - Flatten the saved state dict to REMOVE the wrapper-level 'model.' so newly
            saved checkpoints match the older (short) format.
        - During load, transparently accept either format by stripping a single
            leading 'model.' segment if present.
        - This keeps future checkpoints consistent and silences warnings.
    """
        
    def __init__(self, tokenizer=None, projected_emoji_vectors=None):
        """Optionally pass an already-extended tokenizer and projected emoji vectors.

        projected_emoji_vectors: dict[str, torch.Tensor] (hidden_size) used to
        initialize new embedding rows corresponding to emoji tokens.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(config.foundation_model.model_name)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.foundation_model.tokenizer_name)
        # If tokenizer length differs from base model's embeddings, resize.
        base_embed = self.model.get_input_embeddings()
        if base_embed.num_embeddings < len(self.tokenizer):
            old_num = base_embed.num_embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
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
        lora_core_config = config.training.lora
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

    def forward(self, input_ids, attention_mask):  # type: ignore[override]
        return self.model(input_ids, attention_mask=attention_mask)

    # ----------------------------
    # State dict compatibility API
    # ----------------------------
    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        """Return inner PEFT model weights without the wrapper 'model.' segment.

        Internal module hierarchy:
            self (PhoBERTLoRA)
              └─ self.model (PeftModel)
                   └─ base_model.model.encoder.layer.*

        Default PyTorch traversal would yield keys beginning with 'model.' for the
        wrapper attribute. We strip that so externally they look as if the
        PhoBERTLoRA instance itself were the PEFT model (matching older checkpoints).
        """
        inner = self.model.state_dict(*args, **kwargs)
        # No extra prefixing -> flatten
        return {k: v for k, v in inner.items()}

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load accepting both flattened (preferred) and legacy 'model.'-prefixed keys."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                remapped[k[len('model.'):]] = v
            else:
                remapped[k] = v
        missing, unexpected = self.model.load_state_dict(remapped, strict=strict)
        # Return original-style key names consistent with provided state_dict
        return missing, unexpected

    # PyTorch's recursive loading uses _load_from_state_dict, so we add a shim
    # here to transparently support both naming schemes when loading the full
    # parent module's state_dict.
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
        # Expected current prefixing for inner PEFT model parameters when *saving*:
        #   (flattened)   prefix + base_model.model.encoder.layer.0...
        # Legacy (refactor) included an extra 'model.' immediately after prefix:
        #   prefix + model.base_model.model.encoder.layer.0...
        # We detect legacy keys and clone them into the flattened names so the
        # underlying self.model loader can resolve them.
        legacy_extra = prefix + 'model.'
        # Keys we want for underlying call should NOT retain the wrapper 'model.'
        # Underlying call itself will shift prefix again, so we invoke it with
        # expected_prefix = prefix + 'model.' and remap short keys to that shape.
        # Simpler: just ensure both variants exist with expected_prefix.
        expected_prefix = prefix + 'model.'
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not k.startswith(expected_prefix):
                # Example: k = 'classifier_encoder.base_model.model.encoder.layer.0...'
                tail = k[len(prefix):]
                candidate = expected_prefix + tail
                if candidate not in state_dict:
                    state_dict[candidate] = state_dict[k]
            elif k.startswith(legacy_extra):
                # Already has legacy extra; nothing needed.
                pass
        # Delegate to underlying model (which expects its own parameters with prefix 'model.')
        self.model._load_from_state_dict(
            state_dict,
            expected_prefix,
            local_metadata.get('model', {}),
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )