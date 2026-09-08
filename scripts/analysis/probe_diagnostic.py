#!/usr/bin/env python3
"""Why probe_top1 = 0.538, and what the number is once the sampling is fixed.

    python scripts/analysis/probe_diagnostic.py --config configs/dino_v0.yaml

Runs the probe twice on the SAME checkpoint:

  * `prefix`  -- reproduces `eval_all.py` exactly (`train[:N]`, `valid[:M]`),
  * `sampled` -- the same sizes drawn as a seeded reservoir sample over the split.

On this corpus the prefix arm trains on a single class (Algebra, 100%) because the
splits are written in corpus order, so the probe cannot do anything except predict
the majority class of the eval slice -- 53.86%, which is the number eval.json
reports as both `probe_top1` and `probe_majority_baseline`. The two arms side by
side are the evidence that the reported figure measures the sampling, not the
encoder.

Reports for each arm: top-1, macro-F1, balanced accuracy, majority baseline, the
number of distinct classes the probe ever predicts, per-class support/precision/
recall/F1, and the confusion matrix (written to CSV).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common import (EmbeddingCache, build_label_map_streaming, encode_texts, load_model,
                    setup)

from math_ebt.data.schema import read_jsonl
from math_ebt.eval.probe import train_probe


@dataclass
class ArmResult:
    name: str
    n_train: int
    n_valid: int
    train_classes: int
    valid_classes: int
    train_majority: float
    metrics: dict[str, float] = field(default_factory=dict)
    predicted_classes: int = 0
    balanced_accuracy: float = 0.0
    per_class: list[dict] = field(default_factory=list)
    confusion: np.ndarray | None = None


def _take_prefix(path: str, n: int):
    """train[:n] without loading the rest of the file (read_jsonl is a generator)."""
    out = []
    for d in read_jsonl(path):
        out.append(d)
        if len(out) >= n:
            break
    return out


def _predict(train_z, train_y, valid_z, valid_y, num_classes, cfg, device) -> tuple[dict, np.ndarray]:
    """train_probe for the headline metrics, plus the raw predictions it discards.

    Rejected alternative: changing eval/probe.py to also return predictions. That
    file is on the phase-1 result path and its numbers must stay byte-identical
    across ablation arms; a diagnostic script does not get to touch it. Refitting a
    second probe here costs a few seconds on cached latents.
    """
    metrics = train_probe(
        train_z, train_y, valid_z, valid_y, num_classes,
        cfg.eval.probe_epochs, cfg.eval.probe_lr, device,
    )
    torch.manual_seed(0)
    probe = torch.nn.Linear(train_z.shape[-1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.eval.probe_lr, weight_decay=0.0)
    tz, ty = train_z.to(device), train_y.to(device)
    for _ in range(cfg.eval.probe_epochs):
        perm = torch.randperm(tz.shape[0], device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i : i + 1024]
            loss = torch.nn.functional.cross_entropy(probe(tz[idx]), ty[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = probe(valid_z.to(device)).argmax(-1).cpu().numpy()
    return metrics, pred


def _analyse(pred: np.ndarray, truth: np.ndarray, num_classes: int, names: list[str]):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(truth, pred):
        conf[t, p] += 1
    per_class, recalls = [], []
    for c in range(num_classes):
        tp = int(conf[c, c])
        support = int(conf[c].sum())
        predicted = int(conf[:, c].sum())
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if support:
            recalls.append(rec)
        per_class.append({
            "class": names[c] if c < len(names) else str(c),
            "support": support, "predicted": predicted,
            "precision": round(prec, 4), "recall": round(rec, 4) if support else None,
            "f1": round(f1, 4),
        })
    balanced = float(np.mean(recalls)) if recalls else float("nan")
    return conf, per_class, balanced, int((conf.sum(axis=0) > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="outputs/analysis")
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-valid", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-prefix", action="store_true",
                    help="only run the corrected arm (the prefix arm needs its own "
                         "forward pass, since the cache holds sampled latents)")
    ap.add_argument("--probe-device", default="cpu", choices=["cpu", "auto"],
                    help="where to FIT the probe. Default cpu: the probe is one "
                         "Linear(384, 15) over cached latents, so the accelerator "
                         "buys nothing, and train_probe calls Tensor.bincount, "
                         "which has no MPS kernel in current torch -- fitting on "
                         "the accelerator would need PYTORCH_ENABLE_MPS_FALLBACK=1 "
                         "to avoid a NotImplementedError at the last line. The "
                         "encoder forward passes still run on `auto`.")
    args = ap.parse_args()

    cfg, tok, device = setup(args.config)
    probe_device = device if args.probe_device == "auto" else torch.device("cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_classes = cfg.eval.probe_num_classes
    print(f"device={device} (probe fitted on {probe_device})")
    model = load_model(cfg, args.ckpt, device)
    arms: list[ArmResult] = []

    # ---- arm 1: corrected sampling (reads the cache written by cache_embeddings) --
    ctr = EmbeddingCache.load(Path(args.cache_dir) / "emb_train.npz")
    cva = EmbeddingCache.load(Path(args.cache_dir) / "emb_valid.npz")
    names = ctr.label_names
    ztr, ytr = torch.from_numpy(ctr.embeddings), torch.from_numpy(ctr.labels)
    zva, yva = torch.from_numpy(cva.embeddings), torch.from_numpy(cva.labels)
    metrics, pred = _predict(ztr, ytr, zva, yva, num_classes, cfg, probe_device)
    conf, per_class, balanced, n_pred = _analyse(pred, cva.labels, num_classes, names)
    arms.append(ArmResult(
        name="sampled", n_train=len(ytr), n_valid=len(yva),
        train_classes=len(set(ctr.labels.tolist())), valid_classes=len(set(cva.labels.tolist())),
        train_majority=max(Counter(ctr.labels.tolist()).values()) / len(ytr),
        metrics=metrics, predicted_classes=n_pred, balanced_accuracy=balanced,
        per_class=per_class, confusion=conf,
    ))

    # ---- arm 2: the prefix slicing eval_all.py actually uses --------------------
    if not args.skip_prefix:
        ptr = _take_prefix(f"{cfg.data.splits_dir}/train.jsonl", args.n_train)
        pva = _take_prefix(f"{cfg.data.splits_dir}/valid.jsonl", args.n_valid)
        # Same label map as the corrected arm -- eval_all.py builds it from the FULL
        # train split and only the probe's train/valid slices are prefixed. Holding
        # the map fixed is what makes this a controlled comparison: the single
        # difference between the two arms is prefix vs reservoir sampling.
        pmap, pnames = build_label_map_streaming(
            f"{cfg.data.splits_dir}/train.jsonl", num_classes)
        pother = len(pnames) - 1
        pytr = torch.tensor([pmap.get(d.domain_label, pother) for d in ptr], dtype=torch.long)
        pyva = torch.tensor([pmap.get(d.domain_label, pother) for d in pva], dtype=torch.long)
        pztr = torch.from_numpy(encode_texts(
            model, tok, [d.type_only_text() for d in ptr], cfg.data.max_seq_len,
            device, args.batch_size, desc="prefix train"))
        pzva = torch.from_numpy(encode_texts(
            model, tok, [d.type_only_text() for d in pva], cfg.data.max_seq_len,
            device, args.batch_size, desc="prefix valid"))
        pmetrics, ppred = _predict(pztr, pytr, pzva, pyva, num_classes, cfg, probe_device)
        pconf, pper, pbal, pnpred = _analyse(ppred, pyva.numpy(), num_classes, pnames)
        arms.append(ArmResult(
            name="prefix", n_train=len(pytr), n_valid=len(pyva),
            train_classes=len(set(pytr.tolist())), valid_classes=len(set(pyva.tolist())),
            train_majority=max(Counter(pytr.tolist()).values()) / len(pytr),
            metrics=pmetrics, predicted_classes=pnpred, balanced_accuracy=pbal,
            per_class=pper, confusion=pconf,
        ))

    # ---- report ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'arm':10s} {'train cls':>9s} {'tr maj':>7s} {'top1':>7s} {'macroF1':>8s} "
          f"{'balAcc':>7s} {'baseline':>9s} {'pred cls':>8s}")
    print("-" * 78)
    for a in arms:
        print(f"{a.name:10s} {a.train_classes:9d} {a.train_majority:7.1%} "
              f"{a.metrics['probe_top1']:7.4f} {a.metrics['probe_macro_f1']:8.4f} "
              f"{a.balanced_accuracy:7.4f} {a.metrics['probe_majority_baseline']:9.4f} "
              f"{a.predicted_classes:8d}")
    print("=" * 78)
    for a in arms:
        delta = a.metrics["probe_top1"] - a.metrics["probe_majority_baseline"]
        print(f"\n[{a.name}] top1 - majority_baseline = {delta:+.4f}")
        print(f"{'class':22s} {'support':>8s} {'pred':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
        for r in a.per_class:
            if r["support"] == 0 and r["predicted"] == 0:
                continue
            rec = f"{r['recall']:7.3f}" if r["recall"] is not None else "      -"
            print(f"{r['class']:22s} {r['support']:8d} {r['predicted']:7d} "
                  f"{r['precision']:7.3f} {rec} {r['f1']:7.3f}")
        if a.confusion is not None:
            np.savetxt(out / f"confusion_{a.name}.csv", a.confusion, fmt="%d", delimiter=",")

    payload = [{k: v for k, v in a.__dict__.items() if k != "confusion"} for a in arms]
    (out / "probe_diagnostic.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out / 'probe_diagnostic.json'} and confusion_*.csv")


if __name__ == "__main__":
    main()
