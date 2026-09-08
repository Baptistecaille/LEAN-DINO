#!/usr/bin/env python3
"""Phase 1 training entrypoint."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from math_ebt.config import Config
from math_ebt.tokenization import load_tokenizer
from math_ebt.utils.device import best_device
from math_ebt.data.dataset import Collator, DeclarationDataset, build_label_map
from math_ebt.data.schema import read_jsonl
from math_ebt.models.dino import DINOModel
from math_ebt.training.trainer_dino import DinoTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--view-mode", default=None,
                    help="override for the ablation: certified | naive | dropout_only")
    ap.add_argument("--output-dir", default=None,
                    help="override output_dir, e.g. a Google Drive path on Colab so "
                    "checkpoints (and `last.pt`, which auto-resume reads) survive a "
                    "disconnected runtime instead of living on the ephemeral VM disk")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.view_mode:
        cfg.data.view_mode = args.view_mode
        cfg.project.output_dir = f"{cfg.project.output_dir}_{args.view_mode}"
    if args.output_dir:
        cfg.project.output_dir = args.output_dir
    set_seed(cfg.project.seed)

    train = list(read_jsonl(f"{cfg.data.splits_dir}/train.jsonl"))
    label_map = build_label_map(train, cfg.eval.probe_num_classes)
    tok = load_tokenizer(cfg)

    ds = DeclarationDataset(train, cfg, label_map, seed=cfg.project.seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=Collator(tok, cfg),
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    device = best_device()
    model = DINOModel(cfg)
    n_params = sum(p.numel() for p in model.student_parameters())
    epochs = cfg.training.total_steps * cfg.effective_batch / max(len(ds), 1)
    print(f"view_mode={cfg.data.view_mode}  declarations={len(ds)}  "
          f"student params={n_params/1e6:.1f}M  planned epochs={epochs:.1f}")
    if epochs > 200:
        print("  NOTE: >200 epochs over a small corpus. Watch the train/valid loss gap.")

    trainer = DinoTrainer(cfg, model, loader, device)
    resume_path = trainer.out_dir / "last.pt"
    if resume_path.exists():
        trainer.load(resume_path)
        print(f"resumed from {resume_path} at step {trainer.step}")
    trainer.train()


if __name__ == "__main__":
    main()
