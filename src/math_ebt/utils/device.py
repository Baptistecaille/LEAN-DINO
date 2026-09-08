"""Device selection shared by every training/eval script.

Rejected alternative: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`,
used in earlier drafts of these scripts -- silently falls back to CPU on Apple
Silicon, where MPS is available and meaningfully faster for this model size.
"""
from __future__ import annotations

import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
