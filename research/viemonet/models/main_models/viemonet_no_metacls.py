import torch
import torch.nn as nn
import torch.nn.functional as F

from viemonet.models.submodels.comment_classifier import CommentClassifier
from viemonet.models.submodels.emotion_classifier import EmotionClassifier
from viemonet.models.submodels.meta_classifier import MetaClassifier
from viemonet.config import device, config


class ViemonetModel_No_MetaCLS(nn.Module):
    def __init__(
        self,
        class_weights,
        label_smoothing=0.1,
    ):
        super(ViemonetModel_No_MetaCLS, self).__init__()
        self.comment_classifier = CommentClassifier('phobert', 'cnn')
        self.emotion_classifier = EmotionClassifier()
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
        final_probs = (comment_output['probs'] + emotion_output['probs']) / 2
        final_logits = final_probs.log()

        loss = None
        if labels is not None:
            loss = self.criterion(final_logits, labels)

        return {"loss": loss, "logits": final_logits, "probs": final_probs}