"""Premise selection: query = a theorem's statement, positives = premises it uses.

Also implements the hybrid (BM25 + dense via reciprocal rank fusion). In IR the
hybrid essentially always beats either component; if MathEBT adds nothing to the
hybrid, the dense gain is illusory.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .bm25 import BM25

# Premises that appear in almost every proof and swamp Recall@k.
TRIVIAL_PREMISES = {
    "rfl", "Eq.refl", "Eq.symm", "Eq.trans", "id", "trivial",
    "congrArg", "congrFun", "of_eq_true", "eq_self_iff_true",
}


def filter_premises(premises: list[str]) -> list[str]:
    return [p for p in premises if p.split(".")[-1] not in TRIVIAL_PREMISES
            and p not in TRIVIAL_PREMISES]


def recall_at_k(ranked: list[int], positives: set[int], k: int) -> float:
    if not positives:
        return float("nan")
    return len(set(ranked[:k]) & positives) / len(positives)


def mrr(ranked: list[int], positives: set[int]) -> float:
    for rank, idx in enumerate(ranked, start=1):
        if idx in positives:
            return 1.0 / rank
    return 0.0


def dense_rank(
    query_emb: torch.Tensor, corpus_emb: torch.Tensor, k: int
) -> list[list[int]]:
    """Exhaustive cosine. 200k-300k declarations do not need an ANN index."""
    q = torch.nn.functional.normalize(query_emb, dim=-1)
    c = torch.nn.functional.normalize(corpus_emb, dim=-1)
    scores = q @ c.T
    return scores.topk(k, dim=-1).indices.tolist()


def rrf_fuse(rank_lists: list[list[int]], k: int, const: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (const + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


def evaluate(
    ranked_per_query: list[list[int]], positives_per_query: list[set[int]], ks: list[int]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        vals = [
            recall_at_k(r, p, k)
            for r, p in zip(ranked_per_query, positives_per_query)
            if p
        ]
        out[f"recall@{k}"] = sum(vals) / max(len(vals), 1)
    mrrs = [mrr(r, p) for r, p in zip(ranked_per_query, positives_per_query) if p]
    out["mrr"] = sum(mrrs) / max(len(mrrs), 1)
    return out


def build_bm25_ranks(
    queries: list[str], corpus: list[str], k: int
) -> list[list[int]]:
    print(f"BM25: indexing {len(corpus)} documents...")
    index = BM25(corpus)
    return [index.top_k(q, k) for q in tqdm(queries, desc="BM25 queries")]
