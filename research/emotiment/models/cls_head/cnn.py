import torch
import torch.nn as nn
import torch.nn.functional as F
from emotiment.models.cls_head.base_head import BaseHead
from emotiment.models.phobert_lora import PhoBERTLoRA
from emotiment.config import config

class CNNClassifier(BaseHead):
    _keys_to_ignore_on_save = []
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        ccfg = config.model.cnn
        self.classifier_encoder = PhoBERTLoRA()
        self.kernel_sizes = ccfg.kernel_sizes
        self.classifier_convs = nn.ModuleList([
            nn.Conv1d(in_channels=768, out_channels=ccfg.num_filters, kernel_size=k, padding=0)
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
