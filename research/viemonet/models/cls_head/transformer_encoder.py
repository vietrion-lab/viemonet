import torch
import torch.nn as nn

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager


class TransformerEncoderClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        tfm_cfg = config.model.transformer_encoder
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=tfm_cfg.num_heads,
            dim_feedforward=tfm_cfg.dim_feedforward,
            dropout=tfm_cfg.dropout,
            activation=tfm_cfg.activation,
            batch_first=True,
            norm_first=tfm_cfg.norm_first
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tfm_cfg.num_layers,
            norm=nn.LayerNorm(encoder_dim) if tfm_cfg.final_norm else None
        )
        
        self.classifier_dropout = nn.Dropout(tfm_cfg.dropout)
        self.classifier_out = nn.Linear(encoder_dim, tfm_cfg.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        
        # Create attention mask for transformer (inverted: 0 for valid, -inf for padding)
        # Original mask: 1 for valid tokens, 0 for padding
        # Transformer needs: False for valid, True for masked
        src_key_padding_mask = (attention_mask == 0)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Pool the output (mean pooling over sequence)
        x = self.pool(x, attention_mask)
        
        x = self.classifier_dropout(x)
        logits = self.classifier_out(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}
