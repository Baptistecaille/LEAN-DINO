"""Phase 1 training loop. DINO only -- no energy head, no negatives (see docs/DESIGN.md)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models.dino import DINOModel
from .losses import dino_loss
from .schedulers import ema_at, lr_at, teacher_temp_at, wd_at


def _grad_norm(params) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm() for g in grads])).item()


@torch.no_grad()
def _cos_random_pairs(latent: torch.Tensor) -> float:
    """Cosine similarity of random pairs within the batch. -> 1 means the
    representation is collapsing onto a point, independent of the DINO loss value."""
    n = latent.shape[0]
    if n < 2:
        return 0.0
    perm = torch.randperm(n, device=latent.device)
    a = torch.nn.functional.normalize(latent.float(), dim=-1)
    b = a[perm]
    keep = perm != torch.arange(n, device=latent.device)
    if not keep.any():
        return 0.0
    return (a[keep] * b[keep]).sum(-1).mean().item()


class DinoTrainer:
    def __init__(self, cfg: Config, model: DINOModel, loader: DataLoader, device: torch.device):
        self.cfg = cfg
        self.model = model.to(device)
        self.loader = loader
        self.device = device
        self.step = 0

        self.optimizer = torch.optim.AdamW(
            model.param_groups(cfg.training.weight_decay_start),
            lr=cfg.training.lr,
            betas=(0.9, 0.999),
        )
        self.amp_dtype = (
            torch.bfloat16 if cfg.training.precision == "bf16" else torch.float32
        )
        # device_type must match the model's device, not be hardcoded to "cuda" --
        # bf16 autocast works fine on CPU too, and a hardcoded "cuda" throws on any
        # machine without a GPU (this dev box included).
        self.autocast_device = self.device.type
        self.out_dir = Path(cfg.project.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "metrics.jsonl"

    # ---------------------------------------------------------------- schedules
    def _apply_schedules(self) -> tuple[float, float]:
        t = self.cfg.training
        lr = lr_at(self.step, t.lr, t.warmup_steps, t.total_steps)
        wd = wd_at(self.step, t.weight_decay_start, t.weight_decay_end, t.total_steps)
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = lr
            if group["weight_decay"] > 0:  # group 0 only; no-decay group stays at 0
                group["weight_decay"] = wd
        teacher_temp = teacher_temp_at(
            self.step,
            self.cfg.dino.teacher_temp_start,
            self.cfg.dino.teacher_temp_end,
            self.cfg.dino.teacher_temp_warmup_steps,
        )
        return lr, teacher_temp

    # ------------------------------------------------------------------ forward
    def _forward_views(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        globals_ = [v.to(self.device) for v in batch["globals"]]
        locals_ = [v.to(self.device) for v in batch["locals"]]

        # Globals and locals have different sequence lengths, so they are forwarded
        # separately and the projections concatenated on the view axis.
        student_latents, student_projs = [], []
        for v in globals_ + locals_:
            latent, proj = self.model.forward_student(v.input_ids, v.attention_mask)
            student_latents.append(latent)
            student_projs.append(proj)
        student_proj = torch.stack(student_projs, dim=1)  # [B, V, K]
        student_latent0 = student_latents[0]  # global_view_0, for diagnostics only

        with torch.no_grad():
            teacher_projs = [
                self.model.forward_teacher(v.input_ids, v.attention_mask)[1]
                for v in globals_
            ]
        teacher_proj = torch.stack(teacher_projs, dim=1)  # [B, 2, K]
        return student_proj, teacher_proj, student_latent0

    # --------------------------------------------------------------------- step
    def train(self) -> None:
        self.model.train()
        t = self.cfg.training
        pbar = tqdm(total=t.total_steps, desc="dino")
        data_iter = iter(self.loader)
        t0 = time.time()
        seen = 0

        while self.step < t.total_steps:
            lr, teacher_temp = self._apply_schedules()
            wd = self.optimizer.param_groups[0]["weight_decay"]
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss, diag, cos_rand_batch = 0.0, {}, 0.0

            for _ in range(t.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.loader)
                    batch = next(data_iter)

                with torch.autocast(
                    self.autocast_device, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32
                ):
                    student_proj, teacher_proj, student_latent0 = self._forward_views(batch)
                    loss, diag = dino_loss(
                        student_proj,
                        teacher_proj,
                        self.model.center,
                        self.cfg.dino.student_temp,
                        teacher_temp,
                    )
                (loss / t.grad_accum_steps).backward()
                accum_loss += loss.item() / t.grad_accum_steps
                self.model.update_center(teacher_proj, self.cfg.dino.center_momentum)
                seen += student_proj.shape[0]
                with torch.no_grad():
                    diag["latent_variance"] = student_latent0.float().var(dim=0).mean().item()
                    cos_rand_batch = _cos_random_pairs(student_latent0)

            # Freeze the head's last layer early -- prevents early collapse.
            if self.step < self.cfg.dino.freeze_last_layer_steps:
                self.model.student_head.cancel_last_layer_grad()

            grad_norm_backbone = _grad_norm(self.model.student_backbone.parameters())
            grad_norm_head = _grad_norm(self.model.student_head.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.student_parameters() if p.requires_grad],
                t.grad_clip,
            )
            self.optimizer.step()

            momentum = ema_at(
                self.step,
                self.cfg.dino.ema_momentum_start,
                self.cfg.dino.ema_momentum_end,
                t.total_steps,
            )
            self.model.update_teacher(momentum)

            if self.step % t.log_every == 0:
                self._log(
                    {
                        "step": self.step,
                        "epoch": seen / max(len(self.loader.dataset), 1),
                        "loss": accum_loss,
                        "lr": lr,
                        "weight_decay": wd,
                        "teacher_temp": teacher_temp,
                        "ema_momentum": momentum,
                        "grad_norm": float(grad_norm),
                        "grad_norm_backbone": grad_norm_backbone,
                        "grad_norm_head": grad_norm_head,
                        "cos_random_pairs": cos_rand_batch,
                        "examples_per_sec": seen / max(time.time() - t0, 1e-6),
                        **diag,
                    }
                )
            if self.step > 0 and self.step % t.save_every == 0:
                self.save(f"step_{self.step}.pt")

            self.step += 1
            pbar.update(1)

        pbar.close()
        self.save("last.pt")

    # -------------------------------------------------------------------- utils
    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(self, name: str) -> None:
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),  # includes the center buffer
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.project.run_name,
            },
            self.out_dir / name,
        )

    def load(self, path: Path) -> None:
        """Resume from a checkpoint written by `save`. Restores step, model
        (including the center buffer -- see docs/DESIGN.md invariant 5) and optimizer
        state, so `train()` continues the LR/EMA/temperature schedules from where
        they left off rather than restarting them from step 0."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
