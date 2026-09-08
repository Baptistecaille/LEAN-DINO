"""Shared plumbing for the zero-compute analyses.

These scripts re-analyse an existing checkpoint. They never train the encoder, so
they run on a laptop (MPS or CPU) in minutes. Everything here exists to make that
possible on splits that do not fit in RAM.

Two decisions are load-bearing:

1. **Reservoir sampling, not slicing.** `scripts/eval_all.py` takes `train[:20000]`.
   The splits are written in corpus order (`schema.read_dir` sorts the shard glob),
   so a prefix is an *alphabetical prefix of Mathlib*: measured on this corpus,
   `train[:20000]` is 100% `Algebra` and `valid[:5000]` is 53.86% `Algebra` -- which
   is exactly the `probe_majority_baseline` reported in eval.json. Every sample here
   is a seeded reservoir sample over the whole file instead.
   Rejected alternative: `random.sample(list(read_jsonl(path)), n)`, which is what
   the obvious fix looks like -- it materialises the full split first (train.jsonl is
   2.2GB on disk, ~3.9GB as Declaration objects per the note in configs/dino_v0.yaml)
   and OOMs the machine these scripts are meant to run on.

2. **The seed is part of the protocol.** The same `--seed` must be reused across
   ablation arms, for the same reason eval_all.py fixes its own seed: otherwise
   sampling noise leaks into the C - A comparison that is the actual result.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tokenizers import Tokenizer

from math_ebt.config import Config
from math_ebt.data.schema import Declaration, _to_decl
from math_ebt.models.dino import DINOModel
from math_ebt.tokenization import load_tokenizer
from math_ebt.utils.device import best_device

# --------------------------------------------------------------------------- io

# `"module": "Mathlib.Algebra.Group.Basic"` -- the schema emits it as a flat string,
# so it can be read without parsing the whole record.
_MODULE_RE = re.compile(r'"module"\s*:\s*"([^"]*)"')


def iter_jsonl_lines(path: str | Path) -> Iterator[str]:
    """Stream raw lines, skipping blanks. No parsing."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def reservoir_sample(path: str | Path, n: int, seed: int) -> tuple[list[Declaration], int]:
    """Uniform sample of `n` declarations over the WHOLE file, in one pass.

    Algorithm R over raw JSON *strings*; only the `n` survivors are ever parsed.
    Peak memory is `n` lines (~8.4 KB each on this corpus, vs ~15.5 KB once
    materialised as a Declaration) plus one line, independent of file size -- which
    is what makes this runnable on a laptop against a 2.2GB split.

    Returns (sample, total_lines) so callers do not need a second pass over the
    file just to report the population size.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    total = 0
    for i, line in enumerate(iter_jsonl_lines(path)):
        total = i + 1
        if i < n:
            reservoir.append(line)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = line
    return [_to_decl(json.loads(s)) for s in reservoir], total


def domain_label_of(module: str) -> str:
    """Same rule as Declaration.domain_label, on a raw module string.

    Deliberately duplicated rather than imported: this runs over every line of a
    2.2GB file, so it must not require building a Declaration first.
    """
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def build_label_map_streaming(
    path: str | Path, num_classes: int
) -> tuple[dict[str, int], list[str]]:
    """`data.dataset.build_label_map` over the FULL split, without holding it in RAM.

    eval_all.py builds its label map from the entire train split, so reproducing its
    class indices means counting every line -- but a Counter over domain strings has
    a few hundred entries, where the equivalent list of Declarations is ~3.9GB.
    Returns (mapping, index -> display name).
    """
    counts: Counter[str] = Counter()
    for line in iter_jsonl_lines(path):
        # Cheaper than json.loads on every line; the field is a plain JSON string.
        m = _MODULE_RE.search(line)
        counts[domain_label_of(m.group(1) if m else json.loads(line)["module"])] += 1
    top = [lbl for lbl, _ in counts.most_common(num_classes - 1)]
    mapping = {lbl: i for i, lbl in enumerate(top)}
    other = len(top)
    for lbl in counts:
        mapping.setdefault(lbl, other)
    return mapping, top + ["other"]


# ------------------------------------------------------------------------ model


def load_model(cfg: Config, ckpt_path: str, device: torch.device) -> DINOModel:
    """Rebuild DINOModel and load the checkpoint.

    `load_tokenizer` must already have been called on `cfg`: it rewrites
    `cfg.model.vocab_size` to the tokenizer's real size (32000 -> 33464 on the full
    corpus). Sizing the embedding from the YAML value instead leaves the top ~1.5k
    token ids with no row -- silently wrong on MPS/CPU, which is the platform these
    scripts target.
    """
    model = DINOModel(cfg).to(device)
    # weights_only=True is the torch>=2.6 default and is what we want; fall back only
    # if the checkpoint carries non-tensor payload (e.g. a pickled config object).
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:  # pragma: no cover - depends on how the ckpt was written
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    model.eval()
    return model


def build_untrained_model(cfg: Config, device: torch.device, seed: int = 0) -> DINOModel:
    """Randomly initialised twin of the same architecture.

    This is the control the phase-1 gates lack: a gate that an untrained encoder also
    clears measures the corpus, not the training. Used by view_triviality.py.
    """
    torch.manual_seed(seed)
    model = DINOModel(cfg).to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_texts(
    model: DINOModel,
    tok: Tokenizer,
    texts: list[str],
    max_len: int,
    device: torch.device,
    batch_size: int = 64,
    desc: str = "encode",
) -> np.ndarray:
    """Encode to the pre-projection teacher latent -- the vector used downstream.

    Mirrors `scripts/eval_all.py::encode_texts` exactly (same truncation, same
    padding, same `model.encode`) so numbers here are comparable to eval.json.
    Returns float32 [N, d_model] on CPU.
    """
    from tqdm import tqdm

    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    out: list[torch.Tensor] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc=desc, unit="batch"):
        enc = tok.encode_batch(texts[i : i + batch_size])
        ids = torch.tensor([e.ids for e in enc], dtype=torch.long, device=device)
        mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long, device=device)
        out.append(model.encode(ids, mask).float().cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------------ cache


@dataclass
class EmbeddingCache:
    """One .npz per split slice. Written once, read by every analysis script."""

    embeddings: np.ndarray  # [N, d]
    labels: np.ndarray  # [N] int64, index into label_names
    label_names: list[str]
    decl_names: list[str]
    modules: list[str]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            labels=self.labels,
            label_names=np.array(self.label_names, dtype=object),
            decl_names=np.array(self.decl_names, dtype=object),
            modules=np.array(self.modules, dtype=object),
        )

    @staticmethod
    def load(path: str | Path) -> "EmbeddingCache":
        z = np.load(path, allow_pickle=True)
        return EmbeddingCache(
            embeddings=z["embeddings"],
            labels=z["labels"],
            label_names=list(z["label_names"]),
            decl_names=list(z["decl_names"]),
            modules=list(z["modules"]),
        )


def setup(config_path: str, ckpt: str | None = None) -> tuple[Config, Tokenizer, torch.device]:
    cfg = Config.load(config_path)
    device = best_device()
    tok = load_tokenizer(cfg)  # rewrites cfg.model.vocab_size -- must precede the model
    return cfg, tok, device


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without scipy (docs/DESIGN.md: no new dependency without asking)."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")
