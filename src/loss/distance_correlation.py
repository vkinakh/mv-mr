import torch
import torch.nn as nn


class DistanceCorrelation(nn.Module):
    """Distance correlation loss"""

    def __init__(self):
        super(DistanceCorrelation, self).__init__()

    @staticmethod
    def pairwise_dist(a: torch.Tensor) -> torch.Tensor:
        r = torch.sum(a * a, 1)
        r = torch.reshape(r, [-1, 1])

        d = r - 2 * torch.matmul(a, torch.transpose(a, 0, 1)) + torch.transpose(r, 0, 1)
        d = torch.clamp(d, min=1e-4)
        d = torch.sqrt(d)
        return d

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]

        a = self.pairwise_dist(x)
        b = self.pairwise_dist(z)

        A = a - torch.mean(a, 1) - torch.unsqueeze(torch.mean(a, 0), 1) + torch.mean(a)
        B = b - torch.mean(b, 1) - torch.unsqueeze(torch.mean(b, 0), 1) + torch.mean(b)

        dCovXY = torch.sqrt(torch.mean(A * B) / (n ** 2))
        dVarXX = torch.sqrt(torch.mean(A * A) / (n ** 2))
        dVarYY = torch.sqrt(torch.mean(B * B) / (n ** 2))

        dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
        return dCorXY
