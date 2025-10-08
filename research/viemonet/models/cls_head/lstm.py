import torch.nn as nn

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager


class LSTMClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        lstm_config = config.model.lstm
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        self.classifier_lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_config.dropout,
        )
        self.classifier_dropout = nn.Dropout(lstm_config.dropout)
        self.classifier_out = nn.Linear(lstm_config.hidden_size, lstm_config.out_dim)

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