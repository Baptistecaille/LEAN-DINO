#!/usr/bin/env python3
"""Evaluate a baseline encoder (MLM) by linear probing + retrieval cosine (docs/DESIGN.md
sec 13). No invariance/gates here -- those test the certified-view claim, which the
baseline doesn't make. Compare its numbers against dino_v0's eval.json by hand."""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from math_ebt.config import Config
from math_ebt.tokenization import load_tokenizer
from math_ebt.utils.device import best_device
from math_ebt.data.dataset import MLMCollator, MLMTextDataset, build_label_map
from math_ebt.data.schema import Declaration, read_jsonl
from math_ebt.eval.probe import train_probe
from math_ebt.eval.retrieval import build_bm25_ranks, dense_rank, evaluate, filter_premises
from math_ebt.models.mlm import MLMModel


@torch.no_grad()
def extract_latents_from_texts(model, tok, decls, label_map, max_len, device, bs=64):
    zs, ys = [], []
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    for i in range(0, len(decls), bs):
        chunk = decls[i : i + bs]
        enc = tok.encode_batch([d.type_only_text() for d in chunk])
        ids = torch.tensor([e.ids for e in enc], device=device)
        mask = torch.tensor([e.attention_mask for e in enc], device=device)
        zs.append(model.encode(ids, mask).float().cpu())
        ys.append(torch.tensor([label_map.get(d.domain_label, -1) for d in chunk]))
    return torch.cat(zs), torch.cat(ys)


@torch.no_grad()
def encode_texts(model, tok, texts: list[str], max_len: int, device, bs: int = 128):
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok.encode_batch(texts[i : i + bs])
        ids = torch.tensor([e.ids for e in enc], device=device)
        mask = torch.tensor([e.attention_mask for e in enc], device=device)
        out.append(model.encode(ids, mask).float().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = best_device()
    tok = load_tokenizer(cfg)

    model = MLMModel(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    train = list(read_jsonl(f"{cfg.data.splits_dir}/train.jsonl"))
    valid = list(read_jsonl(f"{cfg.data.splits_dir}/valid.jsonl"))
    test = list(read_jsonl(f"{cfg.data.splits_dir}/test.jsonl"))
    label_map = build_label_map(train, cfg.eval.probe_num_classes)
    metrics: dict[str, float] = {}

    ztr, ytr = extract_latents_from_texts(model, tok, train[:50000], label_map, cfg.data.max_seq_len, device)
    zva, yva = extract_latents_from_texts(model, tok, valid, label_map, cfg.data.max_seq_len, device)
    metrics |= train_probe(
        ztr, ytr, zva, yva, cfg.eval.probe_num_classes, cfg.eval.probe_epochs, cfg.eval.probe_lr, device
    )

    corpus = train + valid + test
    name_to_idx = {d.decl_name: i for i, d in enumerate(corpus)}
    corpus_texts = [d.type_only_text() for d in corpus]
    corpus_emb = encode_texts(model, tok, corpus_texts, cfg.data.max_seq_len, device)

    queries: list[Declaration] = []
    positives = []
    for d in test:
        pos = {name_to_idx[p] for p in filter_premises(d.premises_filtered) if p in name_to_idx}
        if pos:
            queries.append(d)
            positives.append(pos)

    q_emb = encode_texts(model, tok, [d.type_only_text() for d in queries], cfg.data.max_seq_len, device)
    dense = dense_rank(q_emb, corpus_emb, args.top_k)
    lexical = build_bm25_ranks([d.type_implicit for d in queries], [d.type_implicit for d in corpus], args.top_k)

    for tag, ranks in (("dense", dense), ("bm25", lexical)):
        for k, v in evaluate(ranks, positives, cfg.eval.retrieval_top_k).items():
            metrics[f"{tag}_{k}"] = v

    print(json.dumps(metrics, indent=2))
    with open(f"{cfg.project.output_dir}/mlm_eval.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
