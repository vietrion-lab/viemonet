import torch
import torch.nn as nn

from research.viemonet.models.submodels.comment_classifier import CommentClassifier
from viemonet.constant import SOCIAL_DATASET_SIZE
from viemonet.config import device


class ViT5Model(nn.Module):
    def __init__(
        self,
        label_smoothing=0.1,
    ):
        super(ViT5Model, self).__init__()
        self.comment_classifier = CommentClassifier('vit5', 'cnn')
        
        # Handle imbalanced classes with class weights
        total = sum(SOCIAL_DATASET_SIZE)
        class_counts = torch.tensor(SOCIAL_DATASET_SIZE, dtype=torch.float32)
        class_weights = total / (3 * class_counts)
        
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