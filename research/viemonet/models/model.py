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
        
    def _rule_judge(self, comment_output, emo_probs):
        batch_size = comment_output.shape[0]
        device = comment_output.device
        
        comment_label = torch.argmax(comment_output, dim=-1)
        if emo_probs is None:
            return {"logits": comment_output, "probs": comment_output}
        
        # Extract the probs tensor from the dictionary if needed
        emo_probs_tensor = emo_probs['probs'] if isinstance(emo_probs, dict) else emo_probs
        emo_label = torch.argmax(emo_probs_tensor, dim=-1)
        
        # Create a new tensor for logits
        logits = torch.zeros(batch_size, 3, device=device, dtype=comment_output.dtype)
        
        # Process each rule condition for the batch
        for i in range(batch_size):
            emo_l = emo_label[i].item()
            comment_l = comment_label[i].item()
            
            if emo_l == 2:
                if comment_l == 2 or comment_l == 1:
                    logits[i] = torch.tensor([0.0, 0.0, 1.0], device=device)
                elif comment_l == 0:
                    logits[i] = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    logits[i] = comment_output[i].detach()
                    
            elif emo_l == 1:
                if comment_l == 2:
                    logits[i] = torch.tensor([1.0, 0.0, 0.0], device=device)
                elif comment_l == 1 or comment_l == 0:
                    logits[i] = torch.tensor([0.0, 1.0, 0.0], device=device)
                else:
                    logits[i] = comment_output[i].detach()
            elif emo_l == 0:
                if comment_l == 0:
                    logits[i] = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    logits[i] = comment_output[i].detach()
            else:
                logits[i] = comment_output[i].detach()
        
        probs = logits
        return {"logits": logits, "probs": probs}

    def forward(self, ids, attn, emo, labels=None):
        comment_output = self.comment_classifier(ids, attn, labels)
        emo_probs = self.emotion_classifier(emo, device=ids.device)

        final_output = self._rule_judge(comment_output['probs'], emo_probs)

        if labels is not None:
            # Use comment classifier logits for training (has gradients)
            loss = self.criterion(comment_output['logits'], labels).mean()
        else:
            loss = None

        return {"loss": loss, "logits": final_output['logits'], "probs": final_output['probs']}