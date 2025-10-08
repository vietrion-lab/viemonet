import torch.nn as nn

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager

class BiLSTMClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        lcfg = config.model.bilstm
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        self.classifier_lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lcfg.hidden_size,
            num_layers=lcfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lcfg.dropout,
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
