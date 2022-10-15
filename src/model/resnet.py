from typing import Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


class ResnetMultiProj(nn.Module):

    """Resnet with projector"""

    def __init__(self, out_dim: Union[str, int], small_kernel: bool = False):
        super().__init__()

        self.backbone = models.resnet50(pretrained=False, zero_init_residual=True)
        if small_kernel:
            # Change kernel sizes (specific for STL10, CIFAR)
            self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.backbone.fc = nn.Identity()
        self.num_features = 2048

        if isinstance(out_dim, int):
            sizes = [self.num_features, out_dim]
        elif isinstance(out_dim, str):
            sizes = [self.num_features] + list(map(int, out_dim.split('-')))
        elif out_dim is None:
            sizes = []

        if len(sizes) == 0:
            self.projector = nn.Identity()
        else:
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector = nn.Sequential(*layers)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.projector(h)
        return h, z
