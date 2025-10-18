import torch
import torch.nn as nn

from viemonet.config import config
from viemonet.models.submodels.mean_pool import MaskedMeanPool


class BaseHead(nn.Module):
    def __init__(self, foundation_model_name=None):
        super(BaseHead, self).__init__()
        assert foundation_model_name is not None, "Foundation model name must be provided."
        self.foundation_model_name = foundation_model_name
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.model.loss.label_smoothing)
        self.pool = MaskedMeanPool()

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("Forward method not implemented.")