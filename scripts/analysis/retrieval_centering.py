#!/usr/bin/env python3
"""Does removing the corpus-wide shared direction fix top10_jaccard?

    python scripts/analysis/retrieval_centering.py \
        --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt

spectrum.py found cos_rand collapses from 0.129 (raw) to 0.0016 once the single
dominant direction is removed (mean_norm_ratio=0.36, and 0.36**2 = 0.130 -- that one
direction accounts for essentially all of the raw anisotropy). That direction
carries no declaration-specific information: every embedding leans on it about
equally, so it inflates cos_pos without discriminating anything. If it is also what
destabilises retrieval ranks -- a certified view moves an embedding a little in the
direction that actually matters, but that movement is swamped by the shared
component when ranking against ~307k candidates -- removing it before ranking
should raise top10_jaccard.

Unlike spectrum.py / view_triviality.py, this is NOT free: top10_jaccard is defined
against the full retrieval corpus (train+valid+test, ~307k declarations -- exactly
what eval_all.py builds), so this script pays for one corpus-encoding pass, on the
order of 10x the single cache_embeddings.py pass (which only encoded a 30k sample).
Expect tens of minutes on a T4, not the few minutes the earlier scripts took. That
cost is paid once no matter how many centering variants get compared afterward, and
it is the same pass eval_all.py already paid to produce the published eval.json.

Procedure, byte-for-byte eval_all.py where it matters, so the RAW row below is a
reproduction check against the published top10_jaccard=0.256, not a new number:
  1. load train+valid+test in full (not a sample -- this IS the retrieval corpus)
  2. same query_pool sampling (random.Random(seed).sample), same anchor/view text
     builders
  3. same slicing quirk as eval_all.py: invariance embeddings are sliced by COUNT
     (`za[:len(queries)]`), not re-filtered to the same declarations as `queries`
     (the premise-selection subset). Reproduced deliberately rather than silently
     fixed, so the RAW row is a fair comparison to eval.json -- worth fixing in
     eval_all.py itself at some point, just not inside a script whose job is to be
     a faithful baseline.
  4. RAW: rank with the encoder's own embeddings, unmodified
  5. CENTERED: subtract the corpus mean, then rank (dense_rank L2-normalizes
     internally, so this is "remove the shared direction, then re-normalize")
  6. optional, --whiten-dims K: also PCA-whiten to K components (fit on the corpus)

Bonus at no extra encoding cost, since the corpus is already ranked: dense
Recall@10 on the premise-selection queries, raw vs centered, printed next to BM25's
recorded value from the original eval.json -- BM25 doesn't touch embeddings, so
centering cannot change it and it is not worth repaying "usually the slowest step"
(eval_all.py's own comment) just to reprint the same number.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from common import encode_texts, load_model, setup

from math_ebt.data.schema import Declaration, read_jsonl
from math_ebt.eval.invariance import top10_jaccard
from math_ebt.eval.retrieval import evaluate, filter_premises

# From the original run's eval.json. BM25 does not depend on the encoder, so
# centering cannot move it -- printed for reference, not recomputed.
BM25_RECALL_AT_10_FROM_EVAL_JSON = 0.021394191965590195


def dense_rank_chunked(query_emb: torch.Tensor, corpus_emb: torch.Tensor, k: int,
                       chunk: int = 2048, desc: str = "dense_rank") -> list[list[int]]:
    """eval_all.py's helper, same chunking (avoids materialising the full
    n_queries x n_corpus score matrix -- 37GB at 30k x 308k in fp32) -- rewritten
    to operate wherever `query_emb`/`corpus_emb` already live, so callers can put
    both on the accelerator once and rank there instead of on CPU."""
    q = torch.nn.functional.normalize(query_emb, dim=-1)
    c = torch.nn.functional.normalize(corpus_emb, dim=-1)
    out: list[list[int]] = []
    n_chunks = (q.shape[0] + chunk - 1) // chunk
    for i in tqdm(range(0, q.shape[0], chunk), total=n_chunks, desc=desc, unit="chunk"):
        scores = q[i : i + chunk] @ c.T
        out.extend(scores.topk(k, dim=-1).indices.tolist())
    return out


def anchor_text(d: Declaration) -> str:
    return d.type_only_text()


def view_text(d: Declaration) -> str:
    """Byte-for-byte the expression eval_all.py / view_triviality.py use."""
    return f"[PROOF] {d.type_alpha or d.type_explicit or d.type_implicit} [SEP]"


def whiten_basis(corpus: torch.Tensor, k: int):
    """Fit PCA whitening on the corpus: mean, top-k right singular vectors, their
    singular values. `torch.linalg.svd` on a (307k, 384) matrix is seconds, not
    minutes -- 384 is the small dimension, so cost does not scale with corpus size.
    """
    mu = corpus.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(corpus - mu, full_matrices=False)
    return mu, vt[:k].T, s[:k]


def apply_whiten(x: torch.Tensor, mu: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (x - mu) @ v / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--max-queries", type=int, default=2000, help="matches eval_all.py")
    ap.add_argument("--top-k", type=int, default=100, help="matches eval_all.py, for the recall bonus")
    ap.add_argument("--batch-size", type=int, default=256, help="256 is safe on a T4; raise if memory allows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--whiten-dims", type=int, default=None,
                    help="also try full PCA whitening to this many components "
                         "(spectrum.py found ~7-19 dims carry the signal -- try "
                         "e.g. 32 or 64). Omit to skip: centering alone is the "
                         "main comparison this script exists for.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cfg, tok, device = setup(args.config)
    print(f"device={device}")

    print("loading full splits (train+valid+test) -- this IS the retrieval corpus, not a sample...")
    train = list(read_jsonl(f"{cfg.data.splits_dir}/train.jsonl"))
    valid = list(read_jsonl(f"{cfg.data.splits_dir}/valid.jsonl"))
    test = list(read_jsonl(f"{cfg.data.splits_dir}/test.jsonl"))
    corpus = train + valid + test
    name_to_idx = {d.decl_name: i for i, d in enumerate(corpus)}
    print(f"corpus = {len(corpus)} declarations (train={len(train)} valid={len(valid)} test={len(test)})")

    model = load_model(cfg, args.ckpt, device)

    print(f"encoding corpus at batch_size={args.batch_size} -- the one real cost in this script...")
    corpus_emb = torch.from_numpy(encode_texts(
        model, tok, [d.type_only_text() for d in corpus],
        cfg.data.max_seq_len, device, args.batch_size, desc="corpus")).to(device)

    query_pool = test if len(test) <= args.max_queries else rng.sample(test, args.max_queries)
    queries, positives = [], []
    for d in query_pool:
        pos = {name_to_idx[p] for p in filter_premises(d.premises_filtered) if p in name_to_idx}
        if pos:
            queries.append(d)
            positives.append(pos)
    print(f"query_pool={len(query_pool)}, of which {len(queries)} have >=1 in-corpus premise")

    anchors_emb = torch.from_numpy(encode_texts(
        model, tok, [anchor_text(d) for d in query_pool],
        cfg.data.max_seq_len, device, args.batch_size, desc="anchors")).to(device)
    views_emb = torch.from_numpy(encode_texts(
        model, tok, [view_text(d) for d in query_pool],
        cfg.data.max_seq_len, device, args.batch_size, desc="views")).to(device)
    # eval_all.py slices by COUNT here, not by re-filtering to `queries`' own
    # declarations -- reproduced deliberately, see the module docstring.
    za, zv = anchors_emb[: len(queries)], views_emb[: len(queries)]

    q_emb = torch.from_numpy(encode_texts(
        model, tok, [anchor_text(d) for d in queries], cfg.data.max_seq_len,
        device, args.batch_size, desc="recall queries")).to(device)

    def run(label: str, c_emb, a_emb, v_emb, qr_emb):
        ranks_anchor = dense_rank_chunked(a_emb, c_emb, 10, desc=f"{label} anchor ranks")
        ranks_view = dense_rank_chunked(v_emb, c_emb, 10, desc=f"{label} view ranks")
        jac = top10_jaccard(ranks_anchor, ranks_view)
        dense_ranks = dense_rank_chunked(qr_emb, c_emb, args.top_k, desc=f"{label} recall ranks")
        recall = evaluate(dense_ranks, positives, [10, 100])
        return jac, recall

    print("\n" + "=" * 66)
    print(f"{'condition':16s} {'top10_jaccard':>14s} {'dense_recall@10':>17s}")
    print("-" * 66)
    results: dict[str, dict] = {}

    jac_raw, recall_raw = run("raw", corpus_emb, za, zv, q_emb)
    results["raw"] = {"top10_jaccard": jac_raw, **{f"dense_{k}": v for k, v in recall_raw.items()}}
    print(f"{'raw':16s} {jac_raw:14.4f} {recall_raw['recall@10']:17.4f}")

    mu = corpus_emb.mean(0, keepdim=True)
    jac_c, recall_c = run("centered", corpus_emb - mu, za - mu, zv - mu, q_emb - mu)
    results["centered"] = {"top10_jaccard": jac_c, **{f"dense_{k}": v for k, v in recall_c.items()}}
    print(f"{'centered':16s} {jac_c:14.4f} {recall_c['recall@10']:17.4f}")

    if args.whiten_dims:
        mu_w, v_basis, s_vals = whiten_basis(corpus_emb, args.whiten_dims)
        wc = apply_whiten(corpus_emb, mu_w, v_basis, s_vals)
        wa = apply_whiten(za, mu_w, v_basis, s_vals)
        wv = apply_whiten(zv, mu_w, v_basis, s_vals)
        wq = apply_whiten(q_emb, mu_w, v_basis, s_vals)
        jac_w, recall_w = run("whitened", wc, wa, wv, wq)
        tag = f"whitened_{args.whiten_dims}"
        results[tag] = {"top10_jaccard": jac_w, **{f"dense_{k}": v for k, v in recall_w.items()}}
        print(f"{'whiten-' + str(args.whiten_dims):16s} {jac_w:14.4f} {recall_w['recall@10']:17.4f}")

    print("=" * 66)
    print(f"gate threshold top10_jaccard_min = {cfg.gates.top10_jaccard_min}")
    print(f"for reference -- BM25 recall@10 from the original eval.json: "
          f"{BM25_RECALL_AT_10_FROM_EVAL_JSON:.4f} (not recomputed; unaffected by centering)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"seed": args.seed, "n_queries_for_recall": len(queries),
               "bm25_recall_at_10_reference": BM25_RECALL_AT_10_FROM_EVAL_JSON,
               **results}
    (out / "retrieval_centering.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'retrieval_centering.json'}")

    print("\nreading guide:")
    print("  centered jaccard >> raw    -> the shared direction was masking real invariance;")
    print("                                a cheap post-hoc fix, worth using by default")
    print("  centered jaccard ~ raw     -> the shared direction was not the bottleneck;")
    print("                                the residual signal itself is too weak/noisy")
    print("  centered recall@10 > bm25  -> centering alone closes the retrieval gap")
    print("  centered recall@10 < bm25  -> gap survives centering; likely needs more training/ablations")


if __name__ == "__main__":
    main()
