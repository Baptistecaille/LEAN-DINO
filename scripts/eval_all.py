#!/usr/bin/env python3
"""Full phase-1 evaluation: probe, retrieval (+BM25 +hybrid), invariance, gates."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from math_ebt.config import Config
from math_ebt.tokenization import load_tokenizer
from math_ebt.utils.device import best_device
from math_ebt.data.dataset import Collator, DeclarationDataset, build_label_map
from math_ebt.data.schema import Declaration, read_jsonl
from math_ebt.eval import gates as gates_mod
from math_ebt.eval.invariance import (alignment_uniformity, mean_cos_positive,
                                      mean_cos_random, top10_jaccard)
from math_ebt.eval.probe import extract_latents, train_probe
from math_ebt.eval.retrieval import (build_bm25_ranks, dense_rank, evaluate,
                                     filter_premises, rrf_fuse)
from math_ebt.models.dino import DINOModel

_T0 = time.time()


def log(msg: str) -> None:
    """Stage banner with wall-clock elapsed time, so a stalled step is visible
    instead of a silent gap between the two print(json.dumps(...)) calls at the end."""
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


@torch.no_grad()
def encode_texts(model, tok, texts: list[str], max_len: int, device, bs: int = 128, desc: str = "encode"):
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    out = []
    n_batches = (len(texts) + bs - 1) // bs
    for i in tqdm(range(0, len(texts), bs), total=n_batches, desc=desc, unit="batch"):
        enc = tok.encode_batch(texts[i : i + bs])
        ids = torch.tensor([e.ids for e in enc], device=device)
        mask = torch.tensor([e.attention_mask for e in enc], device=device)
        out.append(model.encode(ids, mask).float().cpu())
    return torch.cat(out)


def dense_rank_chunked(query_emb, corpus_emb, k, chunk=2048, desc: str = "dense_rank"):
    """dense_rank in query chunks -- avoids materializing the full
    n_queries x n_corpus score matrix (37GB at 30k x 308k in fp32)."""
    from math_ebt.eval.retrieval import dense_rank
    out = []
    n_chunks = (query_emb.shape[0] + chunk - 1) // chunk
    for i in tqdm(range(0, query_emb.shape[0], chunk), total=n_chunks, desc=desc, unit="chunk"):
        out.extend(dense_rank(query_emb[i : i + chunk], corpus_emb, k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-queries", type=int, default=2000,
                     help="subsample of test used for premise selection + invariance. "
                          "Use the same seed/value across ablation arms so C-A is a "
                          "paired comparison, not noise from different samples.")
    ap.add_argument("--max-probe-train", type=int, default=20000)
    ap.add_argument("--max-probe-valid", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=None,
                     help="override output_dir, mirroring train_dino.py. Needed on "
                          "Colab: metrics.jsonl (read by the stability gate) and "
                          "eval.json (written at the end) otherwise resolve against "
                          "the ephemeral VM disk while the checkpoint lives on Drive, "
                          "and the run dies on FileNotFoundError after all the work.")
    ap.add_argument("--fast", action="store_true",
                     help="debug mode: tiny corpus + tiny queries, ~1min, "
                          "numbers are not meaningful, just checks the script runs")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    log(f"config={args.config} ckpt={args.ckpt} max_queries={args.max_queries} "
        f"max_probe_train={args.max_probe_train} max_probe_valid={args.max_probe_valid} "
        f"seed={args.seed} fast={args.fast}")

    cfg = Config.load(args.config)
    if args.output_dir:
        cfg.project.output_dir = args.output_dir
    device = best_device()
    log(f"device={device} output_dir={cfg.project.output_dir}")
    tok = load_tokenizer(cfg)

    model = DINOModel(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()
    log("model loaded")

    train = list(read_jsonl(f"{cfg.data.splits_dir}/train.jsonl"))
    valid = list(read_jsonl(f"{cfg.data.splits_dir}/valid.jsonl"))
    test = list(read_jsonl(f"{cfg.data.splits_dir}/test.jsonl"))
    log(f"splits loaded: train={len(train)} valid={len(valid)} test={len(test)}")
    label_map = build_label_map(train, cfg.eval.probe_num_classes)
    metrics: dict[str, float] = {}

    if args.fast:
        print("--fast: shrinking corpus too. Numbers are NOT comparable across runs "
              "or to the gate thresholds -- this only checks the script runs end to end.")
        rng.shuffle(train); rng.shuffle(valid); rng.shuffle(test)
        train, valid, test = train[:5000], valid[:1000], test[:500]
        args.max_queries = min(args.max_queries, 200)
        args.max_probe_train = min(args.max_probe_train, 3000)
        args.max_probe_valid = min(args.max_probe_valid, 500)

    # Same subsample (fixed seed) must be reused across ablation arms (certified/
    # naive/dropout_only) -- otherwise sampling noise leaks into C - A, which is the
    # actual quantity being measured.
    # FIX (2026-08-16): was `train[: args.max_probe_train]` / `valid[: args.max_probe_valid]`
    # -- a prefix slice, not a sample. `data/splits/*.jsonl` are written in Mathlib's
    # corpus order (alphabetical-ish by file/namespace), so a prefix is a handful of
    # early namespaces, not a cross-section of the 15 probe classes -- this is what
    # produced the invalid probe_top1=0.538/macro_f1=0.100 result (only 4/15 classes
    # ever appeared in that slice). Reservoir sampling via the same seeded `rng` used
    # for query_pool below, so the subsample is reproducible and identical across the
    # three ablation arms (same seed => same indices), preserving the paired-comparison
    # property the comment above requires.
    probe_train = train if len(train) <= args.max_probe_train else rng.sample(train, args.max_probe_train)
    probe_valid = valid if len(valid) <= args.max_probe_valid else rng.sample(valid, args.max_probe_valid)
    query_pool = test if len(test) <= args.max_queries else rng.sample(test, args.max_queries)
    log(f"subsampled: probe_train={len(probe_train)} probe_valid={len(probe_valid)} "
        f"query_pool={len(query_pool)}")

    # ---- 1. linear probe on frozen latents ---------------------------------
    log("[1/4] linear probe: extracting latents")

    def loader_for(decls: list[Declaration]) -> DataLoader:
        # Workers fork the Declaration list (~15.5 KB/decl) and CPython refcounting
        # breaks copy-on-write, so worker count multiplies RAM. Capped at 2: by this
        # point `corpus`, `corpus_texts` and the BM25 index are already resident.
        ds = DeclarationDataset(decls, cfg, label_map, seed=0)
        return DataLoader(ds, batch_size=64, collate_fn=Collator(tok, cfg),
                          num_workers=min(cfg.data.num_workers, 2))

    ztr, ytr = extract_latents(model, tqdm(loader_for(probe_train), desc="probe train latents"), device)
    zva, yva = extract_latents(model, tqdm(loader_for(probe_valid), desc="probe valid latents"), device)
    log(f"[1/4] linear probe: fitting ({cfg.eval.probe_epochs} epochs on {len(ztr)} latents)")
    metrics |= train_probe(ztr, ytr, zva, yva, cfg.eval.probe_num_classes,
                           cfg.eval.probe_epochs, cfg.eval.probe_lr, device)
    log(f"[1/4] linear probe done: top1={metrics.get('probe_top1'):.4f} "
        f"macro_f1={metrics.get('probe_macro_f1'):.4f} "
        f"majority_baseline={metrics.get('probe_majority_baseline'):.4f}")

    # ---- 2. premise selection ----------------------------------------------
    log("[2/4] premise selection: encoding corpus")
    corpus = train + valid + test
    name_to_idx = {d.decl_name: i for i, d in enumerate(corpus)}
    corpus_texts = [d.type_only_text() for d in corpus]
    corpus_emb = encode_texts(model, tok, corpus_texts, cfg.data.max_seq_len, device,
                              desc=f"corpus ({len(corpus)} decls)")

    queries, positives = [], []
    for d in query_pool:
        pos = {name_to_idx[p] for p in filter_premises(d.premises_filtered)
               if p in name_to_idx}
        if pos:
            queries.append(d)
            positives.append(pos)
    log(f"[2/4] premise-selection queries: {len(queries)} (sampled from {len(test)} test decls)")

    q_emb = encode_texts(model, tok, [d.type_only_text() for d in queries],
                         cfg.data.max_seq_len, device, desc="query embeddings")
    log("[2/4] dense ranking")
    dense = dense_rank_chunked(q_emb, corpus_emb, args.top_k, desc="dense ranks")
    log("[2/4] BM25 ranking (single-core, this is usually the slowest step)")
    lexical = build_bm25_ranks([d.type_implicit for d in queries],
                               [d.type_implicit for d in corpus], args.top_k)
    hybrid = [rrf_fuse([a, b], args.top_k) for a, b in zip(dense, lexical)]

    for tag, ranks in (("dense", dense), ("bm25", lexical), ("hybrid", hybrid)):
        for k, v in evaluate(ranks, positives, cfg.eval.retrieval_top_k).items():
            metrics[f"{tag}_{k}"] = v
    log(f"[2/4] retrieval done: dense_recall@10={metrics.get('dense_recall@10'):.4f} "
        f"bm25_recall@10={metrics.get('bm25_recall@10'):.4f} "
        f"hybrid_recall@10={metrics.get('hybrid_recall@10'):.4f}")

    # ---- 3. invariance to certified views ----------------------------------
    log("[3/4] invariance: encoding anchors + certified views")
    anchors = [d.type_only_text() for d in query_pool]
    views = [f"[PROOF] {d.type_alpha or d.type_explicit or d.type_implicit} [SEP]"
             for d in query_pool]
    za = encode_texts(model, tok, anchors, cfg.data.max_seq_len, device, desc="anchors")
    zv = encode_texts(model, tok, views, cfg.data.max_seq_len, device, desc="certified views")
    metrics["cos_pos"] = mean_cos_positive(za, zv)
    metrics["cos_rand"] = mean_cos_random(za)
    align, unif = alignment_uniformity(za[:2000], zv[:2000])
    metrics["alignment"], metrics["uniformity"] = align, unif

    ranks_anchor = dense_rank_chunked(za[: len(queries)], corpus_emb, 10, desc="anchor ranks")
    ranks_view = dense_rank_chunked(zv[: len(queries)], corpus_emb, 10, desc="view ranks")
    metrics["top10_jaccard"] = top10_jaccard(ranks_anchor, ranks_view)
    log(f"[3/4] invariance done: cos_pos={metrics['cos_pos']:.4f} "
        f"cos_rand={metrics['cos_rand']:.4f} top10_jaccard={metrics['top10_jaccard']:.4f}")

    # ---- 4. gates ------------------------------------------------------------
    log("[4/4] checking gates")
    results = gates_mod.check_gates(metrics, cfg.gates)
    # metrics.jsonl is written by the trainer next to the checkpoint (--ckpt's
    # directory), NOT necessarily cfg.project.output_dir -- those diverge
    # whenever --output-dir is used to redirect eval.json to a different folder
    # than the one being evaluated (e.g. evaluating an intermediate step_N.pt
    # into its own output dir so it doesn't clobber the run's main eval.json).
    # Using cfg.project.output_dir here made the stability gate crash with
    # FileNotFoundError right at the last step, after the full corpus encoding
    # + BM25 pass had already run to completion (hit 2026-08-18 evaluating
    # ablation_naive_freezefix/step_1500.pt into a separate output dir).
    metrics_path = Path(args.ckpt).parent / "metrics.jsonl"
    results.append(gates_mod.no_collapse(str(metrics_path), cfg.gates.min_stable_steps))
    print(json.dumps(metrics, indent=2))
    print()
    print(gates_mod.report(results))

    # Unlike train_dino.py (which mkdir's its out_dir on construction), this
    # script always used to reuse an existing training output_dir, so nothing
    # here ever created cfg.project.output_dir. Exposed 2026-08-18 evaluating a
    # checkpoint into a fresh --output-dir that had never been created by a
    # training run: the whole eval (corpus encoding + BM25) completed and
    # printed successfully, then died on this open() with FileNotFoundError,
    # discarding the result.
    Path(cfg.project.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{cfg.project.output_dir}/eval.json", "w") as fh:
        json.dump({"metrics": metrics,
                   "gates": [r.__dict__ for r in results]}, fh, indent=2)
    log(f"done, wrote {cfg.project.output_dir}/eval.json")


if __name__ == "__main__":
    main()
