"""DINO loss: soft cross-entropy between teacher targets and student predictions."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dino_loss(
    student_proj: torch.Tensor,  # [B, V, K]  V = 2 globals + N locals
    teacher_proj: torch.Tensor,  # [B, 2, K]
    center: torch.Tensor,  # [K]
    student_temp: float,
    teacher_temp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Each teacher global supervises every student view EXCEPT the identical one.

    Returns (loss, diagnostics). The diagnostics are not optional: teacher entropy is
    the only reliable early warning of collapse, and it must be logged from step 0.
    """
    teacher_logits = (teacher_proj.float() - center.view(1, 1, -1)) / teacher_temp
    teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
    student_log_probs = F.log_softmax(student_proj.float() / student_temp, dim=-1)

    n_views = student_proj.shape[1]
    total = student_proj.new_zeros(())
    n_terms = 0
    for t_idx in range(teacher_proj.shape[1]):
        target = teacher_probs[:, t_idx, :]
        for s_idx in range(n_views):
            if s_idx == t_idx:  # skip the identical view
                continue
            total = total - (target * student_log_probs[:, s_idx, :]).sum(-1).mean()
            n_terms += 1
    loss = total / max(n_terms, 1)

    with torch.no_grad():
        ent = -(teacher_probs * torch.log(teacher_probs + 1e-9)).sum(-1)
        diag = {
            "teacher_entropy_mean": ent.mean().item(),
            "teacher_entropy_min": ent.min().item(),
            "teacher_entropy_max": ent.max().item(),
            "teacher_max_prob": teacher_probs.max(dim=-1).values.mean().item(),
            "center_norm": center.norm().item(),
            "student_proj_std": student_proj.float().std().item(),
        }
    return loss, diag
