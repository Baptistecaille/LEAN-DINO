"""Linear probing on the frozen latent. Diagnostic, not the project's headline metric."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.heads import LinearProbe


def train_probe(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    valid_z: torch.Tensor,
    valid_y: torch.Tensor,
    num_classes: int,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or torch.device("cpu")
    probe = LinearProbe(train_z.shape[-1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    train_z, train_y = train_z.to(device), train_y.to(device)
    valid_z, valid_y = valid_z.to(device), valid_y.to(device)

    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(train_z.shape[0], device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i : i + 1024]
            loss = F.cross_entropy(probe(train_z[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(valid_z).argmax(-1)
    top1 = (pred == valid_y).float().mean().item()

    # macro-F1, so that a majority-class-only probe cannot look good
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (valid_y == c)).sum().item()
        fp = ((pred == c) & (valid_y != c)).sum().item()
        fn = ((pred != c) & (valid_y == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    majority = valid_y.bincount(minlength=num_classes).max().item() / len(valid_y)
    return {
        "probe_top1": top1,
        "probe_macro_f1": sum(f1s) / max(len(f1s), 1),
        "probe_majority_baseline": majority,
    }


@torch.no_grad()
def extract_latents(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    zs, ys = [], []
    for batch in loader:
        v = batch["globals"][1].to(device)  # type-only view: matches retrieval
        zs.append(model.encode(v.input_ids, v.attention_mask).float().cpu())
        ys.append(batch["labels"])
    return torch.cat(zs), torch.cat(ys)
