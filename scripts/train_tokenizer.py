#!/usr/bin/env python3
"""Train the BPE tokenizer on the Lean corpus, with frequent constants pre-seeded."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from math_ebt.config import Config
from math_ebt.data.schema import SPECIAL_TOKENS, read_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--top-constants", type=int, default=2000)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    decls = read_dir(cfg.data.raw_dir)

    # Frequent Mathlib constants become atomic tokens rather than being shredded
    # into subwords by BPE.
    counts: Counter[str] = Counter()
    for d in decls:
        counts.update(d.used_constants)
    frequent = [c for c, _ in counts.most_common(args.top_constants)]

    corpus: list[str] = []
    for d in decls:
        corpus += [d.type_implicit, d.type_explicit, d.proof_source]
        corpus += [v for v in (d.type_alpha, d.type_perm_hyp) if v]

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.model.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tok.train_from_iterator((t for t in corpus if t), trainer=trainer)
    tok.add_tokens(frequent)

    out = Path(cfg.data.tokenizer_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))

    assert tok.token_to_id("[PROOF]") is not None
    print(f"saved {out}  vocab={tok.get_vocab_size()}  "
          f"(+{len(frequent)} constants)  [PROOF]={tok.token_to_id('[PROOF]')}")


if __name__ == "__main__":
    main()
