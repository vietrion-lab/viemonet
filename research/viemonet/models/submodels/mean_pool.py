import torch.nn as nn


class MaskedMeanPool(nn.Module):
    def forward(self, H, mask):        # H: (B,L,H), mask: (B,L)
        m = mask.unsqueeze(-1)         # (B,L,1)
        s = (H * m).sum(dim=1)         # (B,H)
        d = m.sum(dim=1).clamp(min=1)  # (B,1)
        return s / d