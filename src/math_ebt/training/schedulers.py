"""Cosine schedules with warmup for lr, weight decay, EMA momentum, teacher temp."""
from __future__ import annotations

import math


def cosine(start: float, end: float, step: int, total: int) -> float:
    if total <= 0:
        return end
    t = min(max(step / total, 0.0), 1.0)
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    return cosine(base_lr, base_lr * 1e-2, step - warmup, max(total - warmup, 1))


def wd_at(step: int, start: float, end: float, total: int) -> float:
    return cosine(start, end, step, total)


def ema_at(step: int, start: float, end: float, total: int) -> float:
    return cosine(start, end, step, total)


def teacher_temp_at(step: int, start: float, end: float, warmup: int) -> float:
    """Linear warmup. Starting at the final temperature is a known source of early
    instability; DINO ramps 0.04 -> 0.07."""
    if step >= warmup:
        return end
    return start + (end - start) * step / max(warmup, 1)
