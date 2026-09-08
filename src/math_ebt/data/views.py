"""View generation -- the heart of the scientific claim.

Three arms, selected by `DataCfg.view_mode`, all producing the same number of views
so that the compute budget is identical across the ablation:

  certified    : Lean-generated views (alpha-renaming, hypothesis permutation,
                 explicit printing, real subterms / tactic slices)
  naive        : text-level augmentations with no semantic guarantee
                 (random span deletion, random span shuffling, random char slices)
  dropout_only : no augmentation at all -- the same text twice, differing only
                 through the model's own dropout (SimCSE). This is the floor.

The certified - dropout_only gap is the result the project exists to produce.

STRUCTURE (frozen, see docs/DESIGN.md):
  global_view_0 = full declaration      (type + proof source)
  global_view_1 = type only             ALWAYS -- retrieval is evaluated type-only
  local_view_*  = short contiguous fragments, max_seq_len_local tokens
"""
from __future__ import annotations

import random

from .schema import Declaration


class ViewGenerator:
    def __init__(self, mode: str, num_local: int = 4, rng: random.Random | None = None):
        if mode not in {"certified", "naive", "dropout_only"}:
            raise ValueError(f"unknown view_mode: {mode}")
        self.mode = mode
        self.num_local = num_local
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------ global
    def global_views(self, d: Declaration) -> tuple[str, str]:
        if self.mode == "dropout_only":
            # Identical text twice; the only difference comes from dropout.
            return d.full_text(), d.full_text()

        v0 = d.full_text()

        if self.mode == "naive":
            return self._naive(v0), self._naive_type_only(d)

        # certified: type-only global, optionally alpha-renamed / permuted / explicit.
        return self._certified_full(d), self._certified_type_only(d)

    def _certified_full(self, d: Declaration) -> str:
        # Transformations apply ON TOP of the fixed view structure, they do not
        # replace it. Rejected alternative: sampling global_view_1 among four
        # candidates -- that made the model see the type-only regime only 25% of the
        # time while evaluation uses it 100% of the time.
        type_str = d.type_implicit
        if d.type_alpha and self.rng.random() < 0.5:
            type_str = d.type_alpha
        elif d.type_perm_hyp and self.rng.random() < 0.5:
            type_str = d.type_perm_hyp
        return f"[PROOF] {type_str} [SEP] {d.proof_source} [SEP]"

    def _certified_type_only(self, d: Declaration) -> str:
        candidates = [d.type_implicit]
        if d.type_explicit:
            candidates.append(d.type_explicit)
        if d.type_alpha:
            candidates.append(d.type_alpha)
        if d.type_perm_hyp:
            candidates.append(d.type_perm_hyp)
        return f"[PROOF] {self.rng.choice(candidates)} [SEP]"

    # ------------------------------------------------------------------- local
    def local_views(self, d: Declaration) -> list[str]:
        if self.mode == "dropout_only":
            return [d.full_text() for _ in range(self.num_local)]
        if self.mode == "naive":
            return [self._naive_fragment(d.full_text()) for _ in range(self.num_local)]
        return self._certified_local(d)

    def _certified_local(self, d: Declaration) -> list[str]:
        """Genuinely local: a real sub-object, not a masked full sequence.

        Pool, in order of preference:
          - a pretty-printed subterm of the type
          - a subset of hypotheses + the goal
          - a contiguous slice of tactic steps
        """
        pool: list[str] = []
        pool += [f"[PROOF] {s} [SEP]" for s in d.type_subterms]

        if d.hypotheses:
            for _ in range(2):
                k = self.rng.randint(1, len(d.hypotheses))
                subset = self.rng.sample(d.hypotheses, k)
                pool.append("[PROOF] " + " [HYP] ".join(subset) + " [SEP]")

        if len(d.proof_steps) >= 2:
            for _ in range(2):
                n = min(len(d.proof_steps), self.rng.randint(1, 3))
                start = self.rng.randint(0, len(d.proof_steps) - n)
                pool.append("[PROOF] " + " ".join(d.proof_steps[start : start + n]) + " [SEP]")

        if not pool:
            # Fallback for declarations with no exploitable structure. Logged by
            # scripts/measure_corpus.py -- if this fires often the extractor is thin.
            pool = [d.type_only_text()]

        return [self.rng.choice(pool) for _ in range(self.num_local)]

    # ------------------------------------------------------------------- naive
    def _naive(self, text: str) -> str:
        toks = text.split()
        if len(toks) < 8:
            return text
        n_del = max(1, int(0.10 * len(toks)))
        idx = set(self.rng.sample(range(1, len(toks)), n_del))
        return " ".join(t for i, t in enumerate(toks) if i not in idx)

    def _naive_type_only(self, d: Declaration) -> str:
        return self._naive(d.type_only_text())

    def _naive_fragment(self, text: str) -> str:
        toks = text.split()
        if len(toks) < 6:
            return text
        n = self.rng.randint(3, max(4, len(toks) // 3))
        start = self.rng.randint(0, len(toks) - n)
        return "[PROOF] " + " ".join(toks[start : start + n]) + " [SEP]"
