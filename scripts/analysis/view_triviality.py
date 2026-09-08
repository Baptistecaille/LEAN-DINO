#!/usr/bin/env python3
"""Is the cos_pos gate measuring the training, or the corpus?

    python scripts/analysis/view_triviality.py \
        --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt

`cos_pos = 0.979` clears its 0.80 threshold comfortably. A gate is only worth its
threshold if something can fail it, so this script attacks it from three sides:

  1. **Text identity.** eval_all builds the certified view as
     `type_alpha or type_explicit or type_implicit`. When `type_alpha` is missing the
     chain falls back to the anchor's own text and the pair is trivially identical.
     Measured share of exactly-identical pairs, plus lexical overlap (Jaccard over
     whitespace tokens and over tokenizer ids) for the rest.
  2. **Untrained control.** The same architecture with random weights, same views.
     Transformers are anisotropic at initialisation; if an untrained encoder already
     scores near the threshold, the gate is not evidence of learned invariance.
  3. **Mismatched control.** cos(anchor_i, view_j) for i != j. The gap between
     cos_pos and this is the part of the score that is actually about the pairing;
     without it, cos_pos is uninterpretable on an anisotropic space.

Finally it correlates per-pair lexical overlap with per-pair cosine: a strong
correlation means the encoder is tracking surface tokens rather than the
alpha-invariant structure the certification is supposed to buy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from common import (build_untrained_model, encode_texts, load_model, reservoir_sample,
                    setup, spearman)

from math_ebt.data.schema import Declaration


def anchor_text(d: Declaration) -> str:
    return d.type_only_text()


def view_text(d: Declaration) -> str:
    """Byte-for-byte the expression scripts/eval_all.py uses for the certified view."""
    return f"[PROOF] {d.type_alpha or d.type_explicit or d.type_implicit} [SEP]"


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (an * bn).sum(1)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def token_jaccard(tok: Tokenizer, x: str, y: str) -> float:
    return _jaccard(set(tok.encode(x).ids), set(tok.encode(y).ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--n", type=int, default=2000, help="matches eval_all's --max-queries")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, tok, device = setup(args.config)
    decls, _ = reservoir_sample(f"{cfg.data.splits_dir}/test.jsonl", args.n, args.seed)
    anchors = [anchor_text(d) for d in decls]
    views = [view_text(d) for d in decls]
    print(f"{len(decls)} test declarations, device={device}")

    # ---- 1. how different are the two texts, before any model is involved -------
    identical = [a == v for a, v in zip(anchors, views)]
    ws = np.array([_jaccard(set(a.split()), set(v.split())) for a, v in zip(anchors, views)])
    tj = np.array([token_jaccard(tok, a, v) for a, v in zip(anchors, views)])
    has_alpha = sum(1 for d in decls if d.type_alpha)
    print(f"\ntype_alpha present:            {has_alpha}/{len(decls)} ({has_alpha / len(decls):.1%})")
    print(f"view text identical to anchor: {sum(identical)}/{len(decls)} ({np.mean(identical):.1%})")
    print(f"whitespace-token Jaccard: mean={ws.mean():.3f} median={np.median(ws):.3f} min={ws.min():.3f}")
    print(f"tokenizer-id     Jaccard: mean={tj.mean():.3f} median={np.median(tj):.3f} min={tj.min():.3f}")

    # ---- 2. trained encoder ------------------------------------------------------
    model = load_model(cfg, args.ckpt, device)
    za = encode_texts(model, tok, anchors, cfg.data.max_seq_len, device, args.batch_size, desc="anchors")
    zv = encode_texts(model, tok, views, cfg.data.max_seq_len, device, args.batch_size, desc="views")
    cos_pair = _cos_rows(za, zv)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(decls))
    # Guard against a fixed point mapping a pair onto itself, which would leak a
    # genuine positive into the mismatched control.
    perm = np.where(perm == np.arange(len(decls)), (perm + 1) % len(decls), perm)
    cos_mismatch = _cos_rows(za, zv[perm])

    # ---- 3. untrained control ----------------------------------------------------
    untrained = build_untrained_model(cfg, device, seed=args.seed)
    ua = encode_texts(untrained, tok, anchors, cfg.data.max_seq_len, device, args.batch_size, desc="untrained anchors")
    uv = encode_texts(untrained, tok, views, cfg.data.max_seq_len, device, args.batch_size, desc="untrained views")
    cos_untrained = _cos_rows(ua, uv)
    cos_untrained_mismatch = _cos_rows(ua, uv[perm])

    thr = cfg.gates.cos_pos_min
    print("\n" + "=" * 66)
    print(f"{'condition':34s} {'mean cos':>10s} {'median':>9s} {'gate':>8s}")
    print("-" * 66)
    for label, vals in (
        ("trained,   matched pairs (cos_pos)", cos_pair),
        ("trained,   mismatched pairs", cos_mismatch),
        ("untrained, matched pairs", cos_untrained),
        ("untrained, mismatched pairs", cos_untrained_mismatch),
    ):
        flag = "PASS" if vals.mean() >= thr else "fail"
        print(f"{label:34s} {vals.mean():10.4f} {np.median(vals):9.4f} {flag:>8s}")
    print("=" * 66)
    print(f"gate threshold cos_pos_min = {thr}")
    print(f"margin over mismatched control: {cos_pair.mean() - cos_mismatch.mean():+.4f}")
    print(f"margin over untrained control:  {cos_pair.mean() - cos_untrained.mean():+.4f}")

    # ---- 4. does cosine just track surface tokens? -------------------------------
    keep = ~np.array(identical)  # identical pairs would inflate both sides
    if keep.sum() > 10:
        pear = float(np.corrcoef(tj[keep], cos_pair[keep])[0, 1])
        spear = spearman(tj[keep], cos_pair[keep])
        print(f"\ncorrelation(token Jaccard, cos) on the {int(keep.sum())} non-identical pairs:"
              f"  pearson={pear:.3f}  spearman={spear:.3f}")
    else:
        pear = spear = float("nan")

    payload = {
        "n": len(decls), "seed": args.seed,
        "type_alpha_present": has_alpha / len(decls),
        "identical_text_share": float(np.mean(identical)),
        "whitespace_jaccard_mean": float(ws.mean()),
        "token_jaccard_mean": float(tj.mean()),
        "cos_pos_trained_matched": float(cos_pair.mean()),
        "cos_trained_mismatched": float(cos_mismatch.mean()),
        "cos_untrained_matched": float(cos_untrained.mean()),
        "cos_untrained_mismatched": float(cos_untrained_mismatch.mean()),
        "gate_threshold": thr,
        "pearson_jaccard_cos": pear,
        "spearman_jaccard_cos": spear,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "view_triviality.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'view_triviality.json'}")

    print("\nreading guide:")
    print("  untrained matched >= threshold -> the gate does not test training at all")
    print("  matched ~ mismatched           -> cos_pos is anisotropy, not invariance")
    print("  high jaccard-cos correlation   -> the encoder tracks tokens, not structure")


if __name__ == "__main__":
    main()
