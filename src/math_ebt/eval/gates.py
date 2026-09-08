"""Phase-1 gates. All must pass before any energy code is written."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import GatesCfg


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def check_gates(metrics: dict[str, float], cfg: GatesCfg) -> list[GateResult]:
    m = metrics
    results = [
        GateResult("cos_positive", m.get("cos_pos", 0) >= cfg.cos_pos_min,
                   m.get("cos_pos", 0), cfg.cos_pos_min,
                   "views of one declaration must be close in absolute terms"),
        GateResult("cos_random", m.get("cos_rand", 1) <= cfg.cos_rand_max,
                   m.get("cos_rand", 1), cfg.cos_rand_max,
                   "distinct declarations must not collapse together"),
        GateResult("top10_jaccard", m.get("top10_jaccard", 0) >= cfg.top10_jaccard_min,
                   m.get("top10_jaccard", 0), cfg.top10_jaccard_min,
                   "swapping a query for a certified view must not change the top-10"),
        GateResult("probe_top1", m.get("probe_top1", 0) >= cfg.probe_top1_min,
                   m.get("probe_top1", 0), cfg.probe_top1_min,
                   "frozen-latent domain classification"),
        GateResult("beats_bm25", m.get("dense_recall@10", 0) > m.get("bm25_recall@10", 1),
                   m.get("dense_recall@10", 0), m.get("bm25_recall@10", 1),
                   "dense retrieval must beat the lexical baseline"),
        GateResult("beats_hybrid_check",
                   m.get("hybrid_recall@10", 0) > m.get("bm25_recall@10", 1),
                   m.get("hybrid_recall@10", 0), m.get("bm25_recall@10", 1),
                   "if the hybrid does not beat BM25, the dense signal adds nothing"),
    ]
    return results


def no_collapse(metrics_path: str | Path, min_steps: int) -> GateResult:
    """Teacher entropy neither collapsed to 0 nor pinned at log(K)."""
    records = [json.loads(l) for l in Path(metrics_path).read_text().splitlines() if l.strip()]
    if not records or records[-1]["step"] < min_steps:
        return GateResult("stability", False, len(records), min_steps, "not enough steps")
    tail = records[-20:]
    ent = [r["teacher_entropy_mean"] for r in tail if "teacher_entropy_mean" in r]
    maxp = [r["teacher_max_prob"] for r in tail if "teacher_max_prob" in r]
    ok = bool(ent) and min(ent) > 0.5 and max(maxp) < 0.95
    return GateResult("stability", ok, min(ent) if ent else 0.0, 0.5,
                      "teacher entropy must be neither degenerate nor uniform")


def report(results: list[GateResult]) -> str:
    lines = ["| gate | value | threshold | pass |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.value:.4f} | {r.threshold:.4f} | "
                     f"{'PASS' if r.passed else 'FAIL'} |")
    if not all(r.passed for r in results):
        lines.append("\n**Phase 2 is blocked.** Do not add the energy head.")
    return "\n".join(lines)
