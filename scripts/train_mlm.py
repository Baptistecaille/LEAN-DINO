#!/usr/bin/env python3
"""MLM baseline entrypoint. Same backbone/tokenizer/max_seq_len as DINO; the
--token-budget flag is what makes the comparison a fair one (see trainer_mlm.py)."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from math_ebt.config import Config
from math_ebt.tokenization import load_tokenizer
from math_ebt.utils.device import best_device
from math_ebt.data.dataset import MLMCollator, MLMTextDataset
from math_ebt.data.schema import read_jsonl
from math_ebt.models.mlm import MLMModel
from math_ebt.training.trainer_mlm import MlmTrainer, special_token_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="total non-pad tokens to train on; defaults to "
        "total_steps * effective_batch * max_seq_len (DINO's global-view token count) "
        "so the two runs see comparable amounts of data",
    )
    args = ap.parse_args()

    cfg = Config.load(args.config)
    set_seed(cfg.project.seed)

    train = list(read_jsonl(f"{cfg.data.splits_dir}/train.jsonl"))
    tok = load_tokenizer(cfg)

    ds = MLMTextDataset(train)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=MLMCollator(tok, cfg.data.max_seq_len),
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    device = best_device()
    model = MLMModel(cfg)
    mask_id = tok.token_to_id("[MASK]")
    assert mask_id is not None, "tokenizer must define [MASK]"

    token_budget = args.token_budget or (
        cfg.training.total_steps * cfg.effective_batch * cfg.data.max_seq_len
    )
    print(f"token_budget={token_budget}  declarations={len(ds)}")

    trainer = MlmTrainer(
        cfg, model, loader, device, mask_id, special_token_ids(tok), token_budget,
    )
    trainer.train()


if __name__ == "__main__":
    main()
