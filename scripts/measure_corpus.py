#!/usr/bin/env python3
"""STEP 1. Run this on one extracted module before writing any training code.

Four numbers decide max_seq_len and view priorities:
  1. type length distribution
  2. proof-source length distribution   <- if this is in the thousands, the extractor
                                           is emitting elaborated terms, not source
  3. pretty-print collision rate        <- distinct ASTs printing identically
  4. share of declarations with usable certified views

One day of work that saves weeks.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from math_ebt.data.schema import read_dir


def pct(xs: list[int]) -> str:
    a = np.array(xs)
    return (f"n={len(a)}  p50={np.percentile(a,50):.0f}  p90={np.percentile(a,90):.0f}  "
            f"p99={np.percentile(a,99):.0f}  max={a.max():.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--budget", type=int, default=512, help="candidate max_seq_len")
    args = ap.parse_args()

    decls = read_dir(args.raw)
    if not decls:
        raise SystemExit(f"no .jsonl found in {args.raw}")

    # whitespace tokens are a ~1.3x underestimate of BPE tokens; good enough to decide
    tl = [len(d.type_implicit.split()) for d in decls]
    pl = [len(d.proof_source.split()) for d in decls]
    fl = [t + p for t, p in zip(tl, pl)]

    print(f"declarations       : {len(decls)}")
    print(f"type length        : {pct(tl)}")
    print(f"proof source length: {pct(pl)}")
    print(f"full length        : {pct(fl)}")
    over = sum(1 for x in fl if x > args.budget)
    print(f"truncated at {args.budget}: {100*over/len(fl):.1f}%")
    if np.percentile(pl, 50) > 500:
        print("\n  WARNING: median proof length > 500 tokens.")
        print("  The extractor is almost certainly emitting elaborated proof TERMS.")
        print("  proof_source must come from declRange source text. See docs/DESIGN.md A1.")

    # 3. pretty-print collisions: same printed type, different AST hash
    by_text: dict[str, set[str]] = {}
    for d in decls:
        by_text.setdefault(d.type_implicit, set()).add(d.type_ast_hash)
    collisions = sum(1 for hs in by_text.values() if len(hs) > 1)
    print(f"\npp collisions      : {collisions} / {len(by_text)} distinct printed types "
          f"({100*collisions/max(len(by_text),1):.2f}%)")

    # 4. availability of certified view material
    have = Counter()
    for d in decls:
        have["alpha"] += bool(d.type_alpha)
        have["perm_hyp"] += bool(d.type_perm_hyp)
        have["explicit"] += bool(d.type_explicit)
        have["subterms"] += bool(d.type_subterms)
        have["proof_steps"] += len(d.proof_steps) >= 2
        have["premises"] += bool(d.premises_filtered)
    print("\ncertified view availability:")
    for k, v in have.items():
        print(f"  {k:12s} {100*v/len(decls):5.1f}%")

    n_no_local = sum(
        1 for d in decls if not d.type_subterms and not d.hypotheses and len(d.proof_steps) < 2
    )
    print(f"\ndeclarations with NO local-view material: {100*n_no_local/len(decls):.1f}%")
    print("(these fall back to type-only locals; if high, the Lean extractor is thin)")


if __name__ == "__main__":
    main()
