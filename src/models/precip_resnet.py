#!/usr/bin/env python3
"""
Minimal precipitation encoder for the first contrastive optimization test.

v1 deliberately uses an ordinary ResNet-18 stem for a clean baseline and
comparison to prior contrastive-image work.  We will treat alternative,
detail-preserving stems as a later architecture ablation.

Input
-----
[B, 1, H, W] transformed 6-h HRRR precipitation.

Outputs
-------
h : [B, 512] backbone representation
z : [B, projection_dim] L2-normalized projection used by InfoNCE
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PrecipResNet18(nn.Module):
    def __init__(
        self,
        *,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()

        net = resnet18(weights=None)

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

