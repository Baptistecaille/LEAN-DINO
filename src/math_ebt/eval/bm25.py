"""Dependency-free BM25. The bar every dense result must clear.

BM25 is strong on premise selection because statements share tokens with their
premises (`add_comm` shows up in theorems about `+`). That is the point: a gate that
BM25 cannot pass is not a gate.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_SPLIT = re.compile(r"[._\s]+|(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """`Nat.add_comm` -> ['nat', 'add', 'comm']; math symbols kept as single tokens."""
    out: list[str] = []
    for chunk in text.split():
        if chunk.isascii() and any(c.isalnum() for c in chunk):
            out.extend(p.lower() for p in _SPLIT.split(chunk) if p)
        else:
            out.append(chunk)
    return out


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75, max_df_frac: float = 0.2):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}
        # Terms in more than `max_df_frac` of the corpus (generic Lean tokens like
        # "eq", "type") get postings lists comparable to the corpus size itself --
        # at ~300k declarations and ~20k queries that's the actual bottleneck in
        # `scores()`, not the ranking math. Their idf is already near-zero from the
        # formula above, so they barely move a ranking; skipping their postings
        # entirely is a safe, standard IR pruning (not an approximation of the
        # score, just dropping terms that were already contributing ~nothing).
        max_df = max_df_frac * n
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                if df[t] <= max_df:
                    self.postings[t].append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def top_k(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        return [i for i, _ in sorted(s.items(), key=lambda kv: -kv[1])[:k]]
