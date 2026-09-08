#!/usr/bin/env python3
"""Quantify the geometry behind cos_pos = 0.98 with top10_jaccard = 0.26.

    python scripts/analysis/spectrum.py --config configs/dino_v0.yaml

Those two numbers are not contradictory, they are a signature: views of one
declaration land almost on top of each other, yet the retrieved neighbourhood is
almost entirely different when you swap one for the other. That happens when the
latent space is dominated by a handful of directions shared by every declaration --
the cosine is then mostly measuring the shared component, and the residual that
actually distinguishes declarations is small enough for retrieval to be noise.

This script turns that reading into numbers, on cached embeddings only (no model
forward pass, seconds on CPU):

  * singular spectrum, explained variance, dims needed for 90/95/99%
  * effective rank exp(H(p)) and participation ratio -- how many directions the
    space really uses, out of d_model
  * the common-direction test: ||mean(z)|| / mean(||z||). Near 1 means every
    embedding is essentially the same vector plus a small residual
  * cos_rand after removing the top-k principal directions. If cos_rand collapses
    toward 0 once 1-5 directions are gone, the anisotropy is concentrated, and
    post-hoc whitening is a free thing to try before spending GPU hours
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import EmbeddingCache


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def mean_cos_random(z: np.ndarray, n_pairs: int, seed: int) -> float:
    """Same estimator as eval/invariance.mean_cos_random, on numpy."""
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    e = _normalize(z)
    return float((e[i[keep]] * e[j[keep]]).sum(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False, help="unused; kept for call-site symmetry")
    ap.add_argument("--cache-dir", default="outputs/analysis")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--n-pairs", type=int, default=100000)
    ap.add_argument("--max-remove", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = EmbeddingCache.load(Path(args.cache_dir) / f"emb_{args.split}.npz")
    z = cache.embeddings.astype(np.float64)
    n, d = z.shape
    print(f"{args.split}: {n} embeddings, d_model={d}")

    # ---- common-direction test -------------------------------------------------
    mu = z.mean(axis=0)
    ratio = float(np.linalg.norm(mu) / np.mean(np.linalg.norm(z, axis=1)))
    print(f"\n||mean(z)|| / mean(||z||) = {ratio:.4f}"
          f"   (0 = centred cloud, 1 = every embedding is the same vector)")

    # ---- spectrum ---------------------------------------------------------------
    zc = z - mu
    sv = np.linalg.svd(zc, compute_uv=False)
    var = sv**2
    p = var / var.sum()
    cum = np.cumsum(p)
    eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-300, None))).sum()))
    part_ratio = float(var.sum() ** 2 / (var**2).sum())
    dims = {q: int(np.searchsorted(cum, q) + 1) for q in (0.50, 0.90, 0.95, 0.99)}

    top5 = float(cum[min(4, d - 1)])
    top10 = float(cum[min(9, d - 1)])
    print(f"\ntop-1 direction explains {p[0]:.2%} of variance; top-5 {top5:.2%}; top-10 {top10:.2%}")
    print(f"dims for 50/90/95/99% of variance: "
          f"{dims[0.50]} / {dims[0.90]} / {dims[0.95]} / {dims[0.99]}  (of {d})")
    print(f"effective rank exp(H) = {eff_rank:.1f}   participation ratio = {part_ratio:.1f}")

    # ---- cos_rand after removing the leading directions -------------------------
    # Two separate effects, reported separately on purpose: subtracting the mean
    # (the shared component) and then projecting out the k highest-variance axes of
    # what is left. Lumping them together would credit direction 1 with the
    # mean-removal, which is usually the bulk of the drop.
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    print(f"\n{'condition':>14s} {'cos_rand':>10s}")
    curve = []
    base = mean_cos_random(z, args.n_pairs, args.seed)
    centred = mean_cos_random(zc, args.n_pairs, args.seed)
    print(f"{'raw':>14s} {base:10.4f}")
    print(f"{'centred':>14s} {centred:10.4f}")
    curve.append({"k": None, "condition": "raw", "cos_rand": base})
    curve.append({"k": 0, "condition": "centred", "cos_rand": centred})
    ks = [k for k in (1, 2, 3, 5, 8, 12, 16, 24, 32) if k <= min(args.max_remove, d - 1)]
    for k in ks:
        basis = vt[:k]
        residual = zc - (zc @ basis.T) @ basis
        val = mean_cos_random(residual, args.n_pairs, args.seed)
        print(f"{f'centred, -{k} dir':>14s} {val:10.4f}")
        curve.append({"k": k, "condition": f"centred_minus_{k}", "cos_rand": val})

    payload = {
        "split": args.split, "n": n, "d_model": d,
        "mean_norm_ratio": ratio,
        "explained_variance_top1": float(p[0]),
        "explained_variance_top5": top5,
        "explained_variance_top10": top10,
        "dims_for_variance": dims,
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "cos_rand_vs_removed_directions": curve,
        "singular_values": [float(v) for v in sv[:64]],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"spectrum_{args.split}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / f'spectrum_{args.split}.json'}")

    print("\nreading guide:")
    print("  effective rank << d_model      -> the encoder uses a small subspace")
    print("  mean_norm_ratio near 1         -> a shared component dominates every vector")
    print("  cos_rand drops sharply with k  -> anisotropy is concentrated; try whitening")
    print("  cos_rand stays high with k     -> the collapse is spread out, not fixable post-hoc")


if __name__ == "__main__":
    main()
