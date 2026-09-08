"""Bidirectional Transformer encoder. No causal mask -- completion is latent."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelCfg


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: ModelCfg, max_seq_len: int):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learned absolute positions. Rejected: RoPE -- fine here, but one more thing
        # to get wrong on a first run and sequences are short.
        self.position_embedding = nn.Embedding(max_seq_len, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent [B, d], hidden_states [B, L, d]).

        The latent is the [PROOF] token at position 0. It is the vector used for every
        downstream task -- always taken BEFORE the DINO projection head.
        """
        b, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(b, length)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.embed_dropout(self.embed_norm(x))
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.final_norm(x)
        return x[:, 0, :], x
