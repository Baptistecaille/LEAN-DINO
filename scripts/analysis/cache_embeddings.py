#!/usr/bin/env python3
"""Encode seeded samples of each split once; every other analysis reads the cache.

    python scripts/analysis/cache_embeddings.py \
        --config configs/dino_v0.yaml \
        --ckpt model/model_20000_step.pt

Writes outputs/analysis/emb_{train,valid,test}.npz. One forward pass, no training:
runs on MPS or CPU. Re-run with the same --seed across ablation arms.

The encoded text is `type_only_text()` -- the type-only regime, matching
`eval.probe.extract_latents` (which takes globals[1]) and the retrieval evaluation.
Encoding anything else here would measure a regime the gates never look at.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import (EmbeddingCache, build_label_map_streaming, encode_texts, load_model,
                    reservoir_sample, setup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--n-train", type=int, default=20000, help="matches eval_all's --max-probe-train")
    ap.add_argument("--n-valid", type=int, default=5000, help="matches eval_all's --max-probe-valid")
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, tok, device = setup(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} splits_dir={cfg.data.splits_dir}")

    model = load_model(cfg, args.ckpt, device)
    print(f"loaded {args.ckpt}")

    # Label map over the FULL train split, exactly as eval_all.py builds it, so the
    # class indices here are the same ones eval.json reports. Only the probe's
    # train/valid *sampling* is corrected -- changing the label map too would make
    # the corrected numbers incomparable to the published ones for a second reason.
    print("building label map (one streaming pass over train.jsonl)...")
    label_map, names = build_label_map_streaming(
        f"{cfg.data.splits_dir}/train.jsonl", cfg.eval.probe_num_classes)
    other_idx = len(names) - 1
    print(f"  {len(label_map)} domains -> {len(names)} classes: {', '.join(names)}")

    label_train, n_train_total = reservoir_sample(
        f"{cfg.data.splits_dir}/train.jsonl", args.n_train, args.seed)

    manifest: dict[str, dict] = {}
    for split, n in (("train", args.n_train), ("valid", args.n_valid), ("test", args.n_test)):
        path = f"{cfg.data.splits_dir}/{split}.jsonl"
        if split == "train":
            decls, total = label_train, n_train_total
        else:
            decls, total = reservoir_sample(path, n, args.seed)
        dist = Counter(d.domain_label for d in decls)
        print(f"\n[{split}] sampled {len(decls)} of {total} lines, {len(dist)} distinct domains")
        for lbl, k in dist.most_common(5):
            print(f"    {lbl:22s} {k:6d} {k / len(decls):7.2%}")

        emb = encode_texts(
            model, tok, [d.type_only_text() for d in decls],
            cfg.data.max_seq_len, device, args.batch_size, desc=f"{split} embeddings",
        )
        cache = EmbeddingCache(
            embeddings=emb,
            labels=np.array([label_map.get(d.domain_label, other_idx) for d in decls], dtype=np.int64),
            label_names=names,
            decl_names=[d.decl_name for d in decls],
            modules=[d.module for d in decls],
        )
        cache.save(out / f"emb_{split}.npz")
        manifest[split] = {
            "sampled": len(decls),
            "population": total,
            "distinct_domains": len(dist),
            "majority_share": max(dist.values()) / len(decls),
            "top_domain": dist.most_common(1)[0][0],
        }
        print(f"    -> {out / f'emb_{split}.npz'}  shape={emb.shape}")

    manifest["_meta"] = {"seed": args.seed, "ckpt": args.ckpt, "label_names": names}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
