import torch
import torch.nn as nn

from viemonet.models.cls_head.base_head import BaseHead
from viemonet.config import config
from viemonet.models.foundation_model_manager import FoundationModelManager


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence models."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, gru_output, attention_mask):
        """
        Args:
            gru_output: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            context: (batch_size, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(gru_output).squeeze(-1)  # (batch_size, seq_len)
        
        # Mask padding tokens - convert mask to same dtype and create proper mask
        mask = (attention_mask == 0).to(scores.dtype)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attn_weights * gru_output, dim=1)  # (batch_size, hidden_size)
        
        return context


class BiGRUAttentionClassifier(BaseHead):
    def __init__(self, foundation_model_name=None):
        super().__init__(foundation_model_name=foundation_model_name)
        gcfg = config.model.bigru_attention
        self.classifier_encoder, _, encoder_dim = FoundationModelManager().get_model(self.foundation_model_name)
        self.classifier_gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=gcfg.hidden_size,
            num_layers=gcfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gcfg.dropout if gcfg.num_layers > 1 else 0,
        )
        self.attention = AttentionLayer(gcfg.hidden_size * 2)
        self.classifier_dropout = nn.Dropout(gcfg.dropout)
        self.classifier_out = nn.Linear(gcfg.hidden_size * 2, gcfg.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        x, _ = self.classifier_gru(x)
        x = self.attention(x, attention_mask)
        x = self.classifier_dropout(x)
        logits = self.classifier_out(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}
