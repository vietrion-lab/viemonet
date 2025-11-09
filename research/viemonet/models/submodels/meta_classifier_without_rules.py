import torch
import torch.nn as nn

from viemonet.config import device


class MetaClassifier_Without_Rules(nn.Module):
    """
    Meta classifier using logistic regression to combine comment sentiment, 
    emotion sentiment, and rule features for final sentiment predictions.
    
    This simpler approach is more interpretable and can be more robust with limited data.
    """
    
    def __init__(self, input_dim=12, output_dim=3):
        super(MetaClassifier_Without_Rules, self).__init__()
        
        # Logistic regression: single linear layer (interpretable weights)
        self.meta_logistic = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, p_comment, p_emotion):
        po = p_comment - p_emotion
        d = torch.abs(po)
        features = torch.cat([p_comment, p_emotion, po, d], dim=-1)

        logits = self.meta_logistic(features)
        probs = logits.softmax(dim=-1)

        return {"logits": logits, "probs": probs}