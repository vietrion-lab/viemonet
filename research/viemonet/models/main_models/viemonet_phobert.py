import torch
import torch.nn as nn

from viemonet.models.submodels.comment_classifier import CommentClassifier
from viemonet.models.submodels.emotion_classifier import EmotionClassifier
from viemonet.models.submodels.meta_classifier import MetaClassifier
from viemonet.config import device, config


class ViemonetModel(nn.Module):
    def __init__(
        self,
        class_weights,
        label_smoothing=0.1,
    ):
        super(ViemonetModel, self).__init__()
        self.comment_classifier = CommentClassifier('phobert', 'lstm')
        self.emotion_classifier = EmotionClassifier()
        self.meta_classifier = MetaClassifier()
        self.softmax = nn.Softmax(dim=-1)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        self.alpha = config.training_setting.multi_task_learning.alpha
        self.beta = config.training_setting.multi_task_learning.beta

    def forward(self, ids, attn, emo, labels=None):
        comment_output = self.comment_classifier(ids, attn)
        # Pass device to emotion_classifier
        emotion_output = self.emotion_classifier(emo)
        meta_cls_output = self.meta_classifier(comment_output['probs'], emotion_output['probs'])
        final_logits = meta_cls_output['logits']
        final_probs = meta_cls_output['probs']
        
        loss = None
        if labels is not None:
            # Multi-task learning with proper weighting
            L_comment = self.criterion(comment_output['logits'], labels)
            L_meta = self.criterion(meta_cls_output['logits'], labels)
            
            # Weighted combination: prioritize meta-classifier
            loss = self.alpha * L_comment + self.beta * L_meta

        return {"loss": loss, "logits": final_logits, "probs": final_probs}