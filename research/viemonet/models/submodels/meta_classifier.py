import torch
import torch.nn as nn

from viemonet.config import device


class MetaClassifier(nn.Module):
    """
    Meta classifier using logistic regression to combine comment sentiment, 
    emotion sentiment, and rule features for final sentiment predictions.
    
    This simpler approach is more interpretable and can be more robust with limited data.
    """
    
    def __init__(self, input_dim=17, output_dim=3):
        """
        Args:
            input_dim: Input dimension (default 9 for 3x3 rule features)
            output_dim: Output dimension (3 classes: positive, neutral, negative)
            device: Device to run the model on (CPU or GPU)
        """
        super(MetaClassifier, self).__init__()
        
        # Logistic regression: single linear layer (interpretable weights)
        self.logistic = nn.Linear(input_dim, output_dim, bias=True)
        
    def _interactive_features(self, p_comment, p_emotion):
        """
        Compute interactive features for the entire batch.
        
        Args:
            p_comment: Comment probabilities (batch_size, 3)
            p_emotion: Emotion probabilities (batch_size, 3)
            
        Returns:
            Interactive features tensor (batch_size, 5)
        """
        comment_l = torch.argmax(p_comment, dim=-1)  # (batch_size,)
        emotion_l = torch.argmax(p_emotion, dim=-1)  # (batch_size,)
        
        # Check if emotion exists: sum of probabilities > 0 means there are emotions
        has_emotion = (p_emotion.sum(dim=-1) > 0).float()  # (batch_size,)

        # Compute features for entire batch using vectorized operations
        same_polarity = (comment_l == emotion_l).float()  # (batch_size,)
        sarcasm = ((comment_l == 2) & ((emotion_l == 0) | (emotion_l == 1))).float()
        irony = ((comment_l == 0) & (emotion_l == 2)).float()
        friendly = ((comment_l == 1) & (emotion_l == 2)).float()

        # Multiply by has_emotion to zero out features when no emotion exists
        same_polarity = same_polarity * has_emotion
        sarcasm = sarcasm * has_emotion
        irony = irony * has_emotion
        friendly = friendly * has_emotion

        # Stack features: (batch_size, 5)
        return torch.stack([same_polarity, sarcasm, irony, friendly, has_emotion], dim=-1)

    def forward(self, p_comment, p_emotion):
        delta = p_comment - p_emotion
        fabs_delta = torch.abs(delta)
        interactive_features = self._interactive_features(p_comment, p_emotion)
        features = torch.cat([p_comment, p_emotion, delta, fabs_delta, interactive_features], dim=-1)

        logits = self.logistic(features)
        probs = logits.softmax(dim=-1)

        return {"logits": logits, "probs": probs}