"""Perceptual distance between two cell images in frozen VGG16 features (LPIPS-style).

Used as an extra training term: the flow model's current estimate of the clean image is
pulled towards the real one in feature space, not only pixel space, so textures and edges
count. VGG16 is deliberately not the encoder used for evaluation (DINOv2), so the probe and
the random-cell comparison stay independent of the training signal. Features from relu1_2,
relu2_2 and relu3_3 are unit-normalised per position and compared by squared distance,
averaged over positions and layers (equal layer weights; no learned calibration).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3 in vgg16.features


class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: LAYERS[-1] + 1].eval()
        for p in feats.parameters():
            p.requires_grad_(False)
        self.feats = feats
        self.register_buffer("mean", _MEAN)
        self.register_buffer("std", _STD)

    def _features(self, x):
        h = ((x + 1) / 2 - self.mean) / self.std  # [-1, 1] -> ImageNet normalisation
        out = []
        for i, layer in enumerate(self.feats):
            h = layer(h)
            if i in LAYERS:
                out.append(h / (h.norm(dim=1, keepdim=True) + 1e-8))
        return out

    def forward(self, a, b):
        """(n, 3, s, s) pairs in [-1, 1] -> per-image distance (n,)."""
        d = 0
        for fa, fb in zip(self._features(a), self._features(b)):
            d = d + ((fa - fb) ** 2).sum(1).mean((1, 2))
        return d / len(LAYERS)
