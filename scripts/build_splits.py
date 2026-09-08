#!/usr/bin/env python3
"""Dedup by AST hash, then split by file. Never by declaration."""
from __future__ import annotations

import argparse

from math_ebt.config import Config
from math_ebt.data.schema import read_dir
from math_ebt.data.splits import dedup_by_hash, save_splits, split_by_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.load(args.config)

    decls = read_dir(cfg.data.raw_dir)
    decls, removed = dedup_by_hash(decls)
    print(f"deduplicated: removed {removed} near-duplicates, {len(decls)} remain")

    splits = split_by_file(decls)
    for name, ds in splits.items():
        files = len({d.file_path for d in ds})
        print(f"  {name:6s} {len(ds):7d} declarations  {files:5d} files")
    save_splits(splits, cfg.data.splits_dir)


if __name__ == "__main__":
    main()
