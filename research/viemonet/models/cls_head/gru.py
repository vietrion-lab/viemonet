import torch.nn as nn

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager

class GRUClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        gru_cfg = config.model.gru
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        self.classifier_gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=gru_cfg.hidden_size,
            num_layers=gru_cfg.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_cfg.dropout,
        )
        self.classifier_dropout = nn.Dropout(gru_cfg.dropout)
        self.classifier_out = nn.Linear(gru_cfg.hidden_size, gru_cfg.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        x, _ = self.classifier_gru(x)
        x = self.pool(x, attention_mask)
        x = self.classifier_dropout(x)
        logits = self.classifier_out(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}
