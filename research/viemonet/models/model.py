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
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        
    def calculate_rule_feature(self, comment_probs, emotion_probs):
        """
        Calculate rule feature matrix from sentiment probabilities.
        
        Args:
            comment_probs: Sentiment probabilities of comment (batch_size, 3)
            emotion_probs: Sentiment probabilities of emotion symbols (batch_size, 3)
            
        Returns:
            Feature matrix X of shape (batch_size, 3, 3)
            where X[b,i,j] = comment_probs[b,i] * emotion_probs[b,j]
        """
        # Reshape for broadcasting: (batch_size, 3, 1) * (batch_size, 1, 3) = (batch_size, 3, 3)
        comment_probs_expanded = comment_probs.unsqueeze(-1)  # (batch_size, 3, 1)
        emotion_probs_expanded = emotion_probs.unsqueeze(-2)  # (batch_size, 1, 3)
        
        # Element-wise multiplication creates the feature matrix
        feature_matrix = comment_probs_expanded * emotion_probs_expanded  # (batch_size, 3, 3)
        
        return feature_matrix

    def forward(self, ids, attn, emo, labels=None):
        comment_output = self.comment_classifier(ids, attn, labels)
        
        if self.method == METHOD[0]:
            emo_mask = torch.tensor(
                [1 if e else 0 for e in emo], 
                dtype=comment_output['probs'].dtype, 
                device=device
            ).unsqueeze(-1)
            
            # Pass device to emotion_classifier
            emotion_output = self.emotion_classifier(emo, device=device)
            rule_feature = self.calculate_rule_feature(comment_output['probs'], emotion_output['probs'])
            meta_cls_output = self.meta_classifier(comment_output['probs'], emotion_output['probs'], rule_feature, labels)
            final_logits = comment_output['logits'] * (1 - emo_mask) + meta_cls_output['logits'] * emo_mask
            final_probs = self.softmax(final_logits)
        else:
            final_logits = comment_output['logits']
            final_probs = comment_output['probs']
        
        loss = None
        if labels is not None:
            L_final = self.criterion(final_logits, labels).mean()
            
            if self.method == METHOD[0]:
                L_comment = self.criterion(comment_output['logits'], labels).mean()
                ce_meta = self.criterion(meta_cls_output['logits'], labels)
                emo_mask_flat = emo_mask.squeeze(-1)
                num_e = emo_mask_flat.sum().clamp(min=1.0)
                L_meta = (emo_mask_flat * ce_meta).sum() / num_e
                
                # Balanced weights to prevent gradient explosion
                loss = L_final + 0.3 * L_comment + 0.7 * L_meta
            else:
                loss = L_final

        return {"loss": loss, "logits": final_logits, "probs": final_probs}