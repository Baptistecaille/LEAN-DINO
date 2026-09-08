"""MLM baseline trainer (docs/DESIGN.md sec 13 / phase-1 baselines: MLM + BM25).

Compute parity with DINO means matching TOKENS seen, not steps: DINO forwards 2
globals + N locals per step, MLM forwards a single sequence. `train()` takes a
token budget rather than reusing `cfg.training.total_steps` directly, so the two
runs can be compared honestly. Rejected alternative: reusing total_steps as-is --
that would let MLM see several times fewer tokens per step, making "MLM baseline"
an unfair floor.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.schema import SPECIAL_TOKENS
from ..models.mlm import MLMModel, mask_tokens
from .schedulers import lr_at


class MlmTrainer:
    def __init__(
        self,
        cfg: Config,
        model: MLMModel,
        loader: DataLoader,
        device: torch.device,
        mask_id: int,
        special_ids: set[int],
        token_budget: int,
        run_name: str = "mlm",
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.loader = loader
        self.device = device
        self.mask_id = mask_id
        self.special_ids = special_ids
        self.token_budget = token_budget
        self.step = 0

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay_start,
        )
        self.amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float32
        self.autocast_device = device.type
        self.out_dir = Path(cfg.project.output_dir) / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "metrics.jsonl"

    def train(self) -> None:
        self.model.train()
        t = self.cfg.training
        data_iter = iter(self.loader)
        t0 = time.time()
        tokens_seen = 0
        seen = 0
        total_steps_estimate = max(self.token_budget // (self.cfg.data.max_seq_len), 1)

        while tokens_seen < self.token_budget:
            lr = lr_at(self.step, t.lr, t.warmup_steps, total_steps_estimate)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)
            batch = batch.to(self.device)

            masked_ids, labels = mask_tokens(
                batch.input_ids,
                batch.attention_mask,
                self.mask_id,
                self.cfg.model.vocab_size,
                self.special_ids,
            )

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                self.autocast_device, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32
            ):
                _, logits = self.model(masked_ids, batch.attention_mask)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
            self.optimizer.step()

            n_tok = int(batch.attention_mask.sum().item())
            tokens_seen += n_tok
            seen += batch.input_ids.shape[0]

            if self.step % t.log_every == 0:
                self._log(
                    {
                        "step": self.step,
                        "epoch": seen / max(len(self.loader.dataset), 1),
                        "loss": loss.item(),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "tokens_seen": tokens_seen,
                        "examples_per_sec": seen / max(time.time() - t0, 1e-6),
                    }
                )
            if self.step > 0 and self.step % t.save_every == 0:
                self.save(f"step_{self.step}.pt")
            self.step += 1

        self.save("last.pt")

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(self, name: str) -> None:
        torch.save(
            {"step": self.step, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            self.out_dir / name,
        )


def special_token_ids(tokenizer) -> set[int]:
    ids = {tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    return {i for i in ids if i is not None}
