import torch.nn as nn
from emotiment.models.cls_head.base_head import BaseHead
from emotiment.models.phobert_lora import PhoBERTLoRA
from emotiment.config import config

class BiLSTMClassifier(BaseHead):
    _keys_to_ignore_on_save = []
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        lcfg = config.model.bilstm
        self.classifier_encoder = PhoBERTLoRA()
        self.classifier_lstm = nn.LSTM(
            input_size=768,
            hidden_size=lcfg.hidden_size,
            num_layers=lcfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lcfg.dropout if lcfg.num_layers > 1 else 0.0,
        )
        self.classifier_dropout = nn.Dropout(lcfg.dropout)
        self.classifier_out = nn.Linear(lcfg.hidden_size * 2, lcfg.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        x, _ = self.classifier_lstm(x)
        x = self.pool(x, attention_mask)
        x = self.classifier_dropout(x)
        logits = self.classifier_out(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}
