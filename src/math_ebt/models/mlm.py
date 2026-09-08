"""MLM baseline. Same backbone, same tokenizer, same max_seq_len as DINO (docs/DESIGN.md
sec 13) -- the only difference is the objective, so a comparison against it means
something.

Head: Linear -> GELU -> LayerNorm -> Linear(vocab). Masking: 15% of non-special,
non-pad tokens; of those, 80% -> [MASK], 10% -> random token, 10% unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .backbone import TransformerBackbone


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MLMModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = TransformerBackbone(cfg.model, cfg.data.max_seq_len)
        self.head = MLMHead(cfg.model.d_model, cfg.model.vocab_size)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent [B,d], token logits [B,L,vocab])."""
        latent, hidden = self.backbone(ids, mask)
        return latent, self.head(hidden)

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Same interface as DINOModel.encode, so eval code can be shared."""
        latent, _ = self.backbone(ids, mask)
        return latent


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_id: int,
    vocab_size: int,
    special_ids: set[int],
    mlm_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard BERT-style masking. Returns (corrupted_ids, labels) where labels is
    -100 (ignored by cross_entropy) everywhere except the masked positions."""
    labels = input_ids.clone()
    prob = torch.full(labels.shape, mlm_prob, device=input_ids.device)

    special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        special |= input_ids == sid
    prob.masked_fill_(special | (attention_mask == 0), 0.0)

    masked = torch.bernoulli(prob).bool()
    labels[~masked] = -100

    out_ids = input_ids.clone()
    replace = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked
    out_ids[replace] = mask_id

    randomize = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
        & masked
        & ~replace
    )
    random_tokens = torch.randint(0, vocab_size, labels.shape, dtype=input_ids.dtype, device=input_ids.device)
    out_ids[randomize] = random_tokens[randomize]
    # remaining 10% of masked positions (masked & ~replace & ~randomize): left unchanged.

    return out_ids, labels
