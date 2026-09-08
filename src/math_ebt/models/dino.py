"""Student/Teacher wrapper with EMA and the centering buffer."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..config import Config
from .backbone import TransformerBackbone
from .heads import DINOHead


class DINOModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        backbone = TransformerBackbone(cfg.model, cfg.data.max_seq_len)
        head = DINOHead(
            in_dim=cfg.model.d_model,
            hidden_dim=cfg.dino.hidden_dim,
            bottleneck_dim=cfg.dino.bottleneck_dim,
            out_dim=cfg.dino.out_dim,
        )
        self.student_backbone = backbone
        self.student_head = head
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = copy.deepcopy(head)
        for p in self.teacher_parameters():
            p.requires_grad_(False)

        # Registered buffer, NOT a free tensor: otherwise it is absent from the
        # checkpoint and resuming silently restarts from a zero center.
        self.register_buffer("center", torch.zeros(cfg.dino.out_dim))

    # ------------------------------------------------------------------ params
    def student_parameters(self):
        yield from self.student_backbone.parameters()
        yield from self.student_head.parameters()

    def teacher_parameters(self):
        yield from self.teacher_backbone.parameters()
        yield from self.teacher_head.parameters()

    def param_groups(self, weight_decay: float) -> list[dict]:
        """No weight decay on biases, norms, or embeddings."""
        decay, no_decay = [], []
        for name, p in list(self.student_backbone.named_parameters()) + list(
            self.student_head.named_parameters()
        ):
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "embedding" in name or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ----------------------------------------------------------------- forward
    def forward_student(self, ids: torch.Tensor, mask: torch.Tensor):
        latent, _ = self.student_backbone(ids, mask)
        return latent, self.student_head(latent)

    @torch.no_grad()
    def forward_teacher(self, ids: torch.Tensor, mask: torch.Tensor):
        latent, _ = self.teacher_backbone(ids, mask)
        return latent, self.teacher_head(latent)

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Latent used for every downstream task: teacher backbone, pre-projection."""
        latent, _ = self.teacher_backbone(ids, mask)
        return latent

    # --------------------------------------------------------------- EMA/center
    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        pairs = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_head, self.teacher_head),
        ]
        for s_mod, t_mod in pairs:
            for p_s, p_t in zip(s_mod.parameters(), t_mod.parameters()):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
            for b_s, b_t in zip(s_mod.buffers(), t_mod.buffers()):
                b_t.data.copy_(b_s.data)

    @torch.no_grad()
    def update_center(self, teacher_proj: torch.Tensor, momentum: float) -> None:
        batch_center = teacher_proj.float().mean(dim=(0, 1))
        self.center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)
