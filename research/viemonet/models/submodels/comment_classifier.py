import torch
import torch.nn as nn

from viemonet.models.head_model_manager import ClassificationHeadManager

class CommentClassifier(nn.Module):
    def __init__(self, foundation_model_name=None, head_model_name=None):
        super(CommentClassifier, self).__init__()
        self.classification_head = ClassificationHeadManager().get_model(head_model_name, foundation_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.classification_head(input_ids, attention_mask, labels=labels)
        probs = torch.softmax(outputs.get("logits"), dim=-1)
        return {"loss": outputs.get("loss"), "logits": outputs.get("logits"), "probs": probs}