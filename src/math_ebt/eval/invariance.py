"""Invariance to certified views.

The naive gate `cos_pos >= 3 * cos_rand` is worthless: in a non-collapsed model
`cos_rand` sits around 0.03-0.10, so the threshold lands at 0.1-0.3 and a useless
model clears it. Two absolute bounds plus an end-to-end check instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_cos_positive(anchor: torch.Tensor, view: torch.Tensor) -> float:
    a = F.normalize(anchor, dim=-1)
    v = F.normalize(view, dim=-1)
    return (a * v).sum(-1).mean().item()


def mean_cos_random(emb: torch.Tensor, n_pairs: int = 10000, seed: int = 0) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = emb.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    keep = i != j
    e = F.normalize(emb, dim=-1).cpu()
    return (e[i[keep]] * e[j[keep]]).sum(-1).mean().item()


def top10_jaccard(
    ranks_anchor: list[list[int]], ranks_view: list[list[int]], k: int = 10
) -> float:
    """The measure that actually matters: does swapping a query for a certified view
    of itself change the retrieved set? If yes, the invariance is not usable."""
    vals = []
    for a, b in zip(ranks_anchor, ranks_view):
        sa, sb = set(a[:k]), set(b[:k])
        if not sa and not sb:
            continue
        vals.append(len(sa & sb) / len(sa | sb))
    return sum(vals) / max(len(vals), 1)


def alignment_uniformity(
    anchor: torch.Tensor, view: torch.Tensor, t: float = 2.0
) -> tuple[float, float]:
    """Wang & Isola (2020). Lower alignment is better; lower uniformity is better."""
    a = F.normalize(anchor, dim=-1)
    v = F.normalize(view, dim=-1)
    alignment = (a - v).norm(dim=-1).pow(2).mean().item()
    sq = torch.cdist(a, a).pow(2)
    n = a.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=a.device)
    uniformity = sq[off].mul(-t).exp().mean().log().item()
    return alignment, uniformity
