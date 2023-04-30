import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling Kullback-Leibler Divergence loss"""
    def __init__(self, T: float):
        """
        Args:
            T: temperature
        """

        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss
