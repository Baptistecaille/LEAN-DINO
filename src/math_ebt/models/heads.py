"""DINO projection head.

Structure is load-bearing (see docs/DESIGN.md invariant 2):
    MLP -> bottleneck(256) -> L2 normalize -> weight-normed Linear(256, K)
with the weight-norm gain g fixed at 1 and NOT trained.

Dropping the bottleneck + normalization is the classic way to get a run that collapses
between 5k and 20k steps while the loss curve still looks like it is learning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm

    _NEW_API = True
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm as _weight_norm

    _NEW_API = False


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 4096,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = _weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self._freeze_gain()
        self.apply(self._init)

    def _freeze_gain(self) -> None:
        if _NEW_API:
            g = self.last_layer.parametrizations.weight.original0
        else:  # pragma: no cover
            g = self.last_layer.weight_g
        with torch.no_grad():
            g.fill_(1.0)
        g.requires_grad_(False)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def cancel_last_layer_grad(self) -> None:
        """Called every step during the first `freeze_last_layer_steps` steps."""
        for p in self.last_layer.parameters():
            p.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class LinearProbe(nn.Module):
    """Frozen-latent classifier. Backbone must be in eval() and no_grad()."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
