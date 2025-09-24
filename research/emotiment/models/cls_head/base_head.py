import torch
import torch.nn as nn

from emotiment.config import config
from emotiment.models.mean_pool import MaskedMeanPool
from emotiment.training.decorators import log_step


class BaseHead(nn.Module):
    def __init__(self):
        super(BaseHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.model.loss.label_smoothing)
        self.pool = MaskedMeanPool()

    @log_step
    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("Forward method not implemented.")