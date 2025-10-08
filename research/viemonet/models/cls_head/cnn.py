import torch
import torch.nn as nn
import torch.nn.functional as F

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager

class CNNClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        ccfg = config.model.cnn
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        self.kernel_sizes = ccfg.kernel_sizes
        self.classifier_convs = nn.ModuleList([
            nn.Conv1d(in_channels=encoder_dim, out_channels=ccfg.num_filters, kernel_size=k, padding=0)
            for k in self.kernel_sizes
        ])
        self.classifier_dropout = nn.Dropout(ccfg.dropout)
        self.classifier_out = nn.Linear(ccfg.num_filters * len(self.kernel_sizes), ccfg.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state  # (B,L,H)
        x = x.transpose(1, 2)  # (B,H,L)
        feats = []
        for conv in self.classifier_convs:
            c = F.relu(conv(x))  # (B,F,L')
            c = F.max_pool1d(c, kernel_size=c.shape[-1])  # (B,F,1)
            feats.append(c.squeeze(-1))  # (B,F)
        x = torch.cat(feats, dim=-1)
        x = self.classifier_dropout(x)
        logits = self.classifier_out(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}
