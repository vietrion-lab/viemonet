import torch
import torch.nn as nn

from viemonet.models.submodels.comment_classifier import CommentClassifier
from viemonet.config import device


class PhoBERTModel(nn.Module):
    def __init__(
        self,
        class_weights,
        label_smoothing=0.1,
    ):
        super(PhoBERTModel, self).__init__()
        self.comment_classifier = CommentClassifier('phobert', 'cnn')
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing,
            reduction='mean'
        )

    def forward(self, ids, attn, labels=None):
        comment_output = self.comment_classifier(ids, attn)
        
        loss = None
        if labels is not None:
            loss = self.criterion(comment_output['logits'], labels)
            
        return {"loss": loss, "logits": comment_output['logits'], "probs": comment_output['probs']}