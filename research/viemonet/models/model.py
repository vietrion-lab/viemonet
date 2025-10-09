import torch
import torch.nn as nn

from viemonet.models.comment_classifier import CommentClassifier
from viemonet.models.emotion_classifier import EmotionClassifier
from viemonet.models.meta_classifier import MetaClassifier
from viemonet.constant import METHOD
from viemonet.config import device


class ViemonetModel(nn.Module):
    def __init__(
        self,
        method,
        foundation_model_name=None,
        head_model_name=None,
        label_smoothing=0.1,
    ):
        super(ViemonetModel, self).__init__()
        self.method = method
        self.comment_classifier = CommentClassifier(foundation_model_name, head_model_name)
        self.emotion_classifier = EmotionClassifier() if method == METHOD[0] else None
        self.meta_classifier = MetaClassifier() if method == METHOD[0] else None
        self.softmax = nn.Softmax(dim=-1)
        
        # Class weights for imbalanced dataset
        # negative:neutral:positive = 2766:1494:1288
        # Using sklearn-style balanced weights: n_samples / (n_classes * n_samples_per_class)
        total = 2766 + 1494 + 1288  # 5548
        class_counts = torch.tensor([2766.0, 1494.0, 1288.0])  # negative, neutral, positive
        class_weights = total / (3 * class_counts)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing,
            reduction='mean'  # Changed from 'none' to 'mean'
        )

    def forward(self, ids, attn, emo, labels=None):
        comment_output = self.comment_classifier(ids, attn)
        
        if self.method == METHOD[0]:
            # Pass device to emotion_classifier
            emotion_output = self.emotion_classifier(emo)
            meta_cls_output = self.meta_classifier(comment_output['probs'], emotion_output['probs'])
            final_logits = meta_cls_output['logits']
            final_probs = meta_cls_output['probs']
        else:
            final_logits = comment_output['logits']
            final_probs = comment_output['probs']
        
        loss = None
        if labels is not None:
            if self.method == METHOD[0]:
                # Multi-task learning with proper weighting
                L_comment = self.criterion(comment_output['logits'], labels)
                L_meta = self.criterion(meta_cls_output['logits'], labels)
                
                # Weighted combination: prioritize meta-classifier
                loss = 0.3 * L_comment + 0.7 * L_meta
            else:
                loss = self.criterion(final_logits, labels)

        return {"loss": loss, "logits": final_logits, "probs": final_probs}