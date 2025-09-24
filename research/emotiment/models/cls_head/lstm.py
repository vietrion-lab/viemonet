import torch.nn as nn

from emotiment.models.cls_head.base_head import BaseHead
from emotiment.models.phobert_lora import PhoBERTLoRA
from emotiment.config import config


class LSTMClassifier(BaseHead):
    # Provide HuggingFace-compatible attributes to avoid Trainer attribute errors
    _keys_to_ignore_on_save = []
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        lstm_config = config.model.lstm
        # All head modules use 'classifier' prefix so optimizer can pick them up
        self.classifier_encoder = PhoBERTLoRA()
        self.classifier_lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier_out = nn.Linear(lstm_config.hidden_size, lstm_config.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        # Align parameter names with Trainer expectations: input_ids, attention_mask, labels
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        x, _ = self.classifier_lstm(x)
        x = self.pool(x, attention_mask)
        x = self.classifier_out(x)

        loss = None
        if labels is not None:
            loss = self.criterion(x, labels)
        return {"loss": loss, "logits": x}