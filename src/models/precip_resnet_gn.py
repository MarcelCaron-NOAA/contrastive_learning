#!/usr/bin/env python3
"""
GroupNorm variant of the precipitation ResNet-18 encoder.

This keeps the v1 architecture fixed except for normalization:
    BatchNorm2d -> GroupNorm(32 groups)

There are no running means/variances, so train() and eval() use the same
normalization rule.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


def group_norm_32(num_channels: int) -> nn.GroupNorm:
    if num_channels % 32 != 0:
        raise ValueError(
            f"Expected channel count divisible by 32; got {num_channels}"
        )
    return nn.GroupNorm(32, num_channels)


class PrecipResNet18GN(nn.Module):
    def __init__(
        self,
        *,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()

        net = resnet18(
            weights=None,
            norm_layer=group_norm_32,
        )

        # Native precipitation is one channel, not RGB.
        net.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        backbone_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = net
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

        self.backbone_dim = backbone_dim
        self.projection_dim = projection_dim

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=-1)
        return h, z

