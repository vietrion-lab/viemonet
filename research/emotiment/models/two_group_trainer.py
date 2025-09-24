from transformers import Trainer
try:
    from transformers.optimization import get_linear_schedule_with_warmup
except ImportError:  # fallback (older versions)
    from transformers import get_linear_schedule_with_warmup  # type: ignore
from torch.optim import AdamW
from emotiment.training.decorators import log_step

from emotiment.config import config


def build_param_groups_for_lora_and_classifier(model, lora_lr, head_lr, weight_decay):
    no_decay_layers = ["bias", "LayerNorm.weight"]
    param_groups = []
    
    def _wd(name):
        return 0.0 if any(nd in name for nd in no_decay_layers) else weight_decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            param_groups.append({
                "params": [param],
                "lr": lora_lr,
                "weight_decay": _wd(name),
            })
        elif name.startswith('classifier'):
            param_groups.append({
                "params": [param],
                "lr": head_lr,
                "weight_decay": _wd(name),
            })
        else:
            # Optional: skip other params (frozen) or add with default lr_lora
            continue
    return param_groups
    

class TwoGroupTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Future-proof: transformers >=5 will remove `tokenizer` in favor of `processing_class`.
        # We try to map automatically while staying compatible with older versions.
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            _tok = kwargs.pop('tokenizer')
            try:
                # Preferred new API
                super().__init__(*args, processing_class=_tok, **kwargs)
            except TypeError:
                # Older transformers version without processing_class param
                super().__init__(*args, tokenizer=_tok, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self.custom_config = config.training

    @log_step
    def create_optimizer(self):
        if self.optimizer is None:
            groups = build_param_groups_for_lora_and_classifier(
                self.model,
                lora_lr=self.custom_config.lora_lr,
                head_lr=self.custom_config.head_lr,
                weight_decay=self.custom_config.weight_decay,
            )
            self.optimizer = AdamW(groups, eps=1e-8)
        return self.optimizer
    
    @log_step
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # HF Trainer may pass optimizer explicitly; prefer provided one
        if optimizer is not None:
            self.optimizer = optimizer
        if self.lr_scheduler is None:
            warmup_steps = int(self.custom_config.warmup_ratio * num_training_steps)
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
            )
        return self.lr_scheduler

