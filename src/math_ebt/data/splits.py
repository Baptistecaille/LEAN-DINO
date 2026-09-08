"""Split by FILE, dedup by AST hash BEFORE splitting.

Splitting by declaration leaks: `foo` and `foo'` in the same file are near-duplicates,
and a test declaration's views must never appear in training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import Declaration


def dedup_by_hash(decls: list[Declaration]) -> tuple[list[Declaration], int]:
    seen: set[str] = set()
    kept: list[Declaration] = []
    for d in decls:
        key = d.ast_hash or d.type_ast_hash
        if not key:
            kept.append(d)  # no hash available; keep and let measure_corpus flag it
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(d)
    return kept, len(decls) - len(kept)


def _bucket(file_path: str, n: int = 100) -> int:
    h = hashlib.sha256(file_path.encode()).hexdigest()
    return int(h[:8], 16) % n


def split_by_file(
    decls: list[Declaration], valid_frac: float = 0.05, test_frac: float = 0.10
) -> dict[str, list[Declaration]]:
    v_cut = int(valid_frac * 100)
    t_cut = v_cut + int(test_frac * 100)
    out: dict[str, list[Declaration]] = {"train": [], "valid": [], "test": []}
    for d in decls:
        b = _bucket(d.file_path)
        split = "valid" if b < v_cut else "test" if b < t_cut else "train"
        out[split].append(d)
    return out


def save_splits(splits: dict[str, list[Declaration]], out_dir: str | Path) -> None:
    from dataclasses import asdict

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, decls in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for d in decls:
                fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
