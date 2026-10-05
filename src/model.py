"""Saree design encoder.

MobileNetV3-Small is the backbone because the brief gives extra credit for
a lean model and the free Kaggle GPU is the expected machine. ImageNet
weights are a starting point only. The projection is trained so that
cosine similarity means "same motif", not "same color".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class SareeEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = net.classifier[0].in_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled ImageNet features, L2-normalized. The projection is not used.

        This is the untrained baseline. A random projection would make the
        comparison about an untrained head, not about ImageNet features.
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.normalize(x, dim=-1)
