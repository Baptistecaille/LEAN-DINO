"""Dataset + collator. Emits 2 global views and N local views per declaration."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from ..config import Config
from .schema import Declaration
from .views import ViewGenerator


@dataclass
class EncodedViews:
    """Padded batch of one view slot. Shapes: [B, L]."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device: torch.device) -> "EncodedViews":
        return EncodedViews(self.input_ids.to(device), self.attention_mask.to(device))


class DeclarationDataset(Dataset):
    def __init__(
        self,
        decls: list[Declaration],
        cfg: Config,
        label_map: dict[str, int],
        seed: int = 0,
    ):
        self.decls = decls
        self.cfg = cfg
        self.label_map = label_map
        # The map is built from the TRAIN split, so a valid/test declaration can
        # carry a domain train never saw. It belongs in the `other` bucket, which
        # build_label_map puts at the highest index. The previous fallback of -1
        # was silent poison: cross_entropy and bincount reject it, and where they
        # don't, a -1 label simply never matches a prediction, quietly depressing
        # probe accuracy instead of failing.
        self.other_label = max(label_map.values(), default=0)
        self.gen = ViewGenerator(
            mode=cfg.data.view_mode,
            num_local=cfg.data.num_local_views,
            rng=random.Random(seed),
        )

    def __len__(self) -> int:
        return len(self.decls)

    def __getitem__(self, i: int) -> dict:
        d = self.decls[i]
        g0, g1 = self.gen.global_views(d)
        locals_ = self.gen.local_views(d)
        return {
            "decl_name": d.decl_name,
            "global_texts": [g0, g1],
            "local_texts": locals_,
            "label": self.label_map.get(d.domain_label, self.other_label),
        }


class Collator:
    """Tokenizes and pads. Globals and locals have DIFFERENT max lengths on purpose."""

    def __init__(self, tokenizer: Tokenizer, cfg: Config):
        self.tok = tokenizer
        self.max_global = cfg.data.max_seq_len
        self.max_local = cfg.data.max_seq_len_local
        self.pad_id = tokenizer.token_to_id("[PAD]")
        assert self.pad_id is not None, "tokenizer must define [PAD]"

    def _encode(self, texts: list[str], max_len: int) -> EncodedViews:
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")
        encs = self.tok.encode_batch(texts)
        ids = torch.tensor([e.ids for e in encs], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in encs], dtype=torch.long)
        return EncodedViews(ids, mask)

    def __call__(self, batch: list[dict]) -> dict:
        n_local = len(batch[0]["local_texts"])
        globals_ = [
            self._encode([b["global_texts"][k] for b in batch], self.max_global)
            for k in range(2)
        ]
        locals_ = [
            self._encode([b["local_texts"][k] for b in batch], self.max_local)
            for k in range(n_local)
        ]
        return {
            "globals": globals_,
            "locals": locals_,
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "decl_names": [b["decl_name"] for b in batch],
        }


class MLMTextDataset(Dataset):
    """Full-declaration text only -- the MLM baseline has no certified-view claim to
    test, so it does not go through ViewGenerator."""

    def __init__(self, decls: list[Declaration]):
        self.decls = decls

    def __len__(self) -> int:
        return len(self.decls)

    def __getitem__(self, i: int) -> str:
        return self.decls[i].full_text()


class MLMCollator:
    """Tokenizes and pads only. Masking happens in the trainer, on tensors, so it can
    use the model's vocab_size and mask_id without threading them through here."""

    def __init__(self, tokenizer: Tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")
        assert self.pad_id is not None, "tokenizer must define [PAD]"

    def __call__(self, texts: list[str]) -> EncodedViews:
        self.tok.enable_truncation(max_length=self.max_len)
        self.tok.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")
        encs = self.tok.encode_batch(texts)
        ids = torch.tensor([e.ids for e in encs], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in encs], dtype=torch.long)
        return EncodedViews(ids, mask)


def build_label_map(decls: list[Declaration], num_classes: int) -> dict[str, int]:
    """Keep the `num_classes - 1` most frequent domains, bucket the rest as `other`."""
    from collections import Counter

    counts = Counter(d.domain_label for d in decls)
    top = [lbl for lbl, _ in counts.most_common(num_classes - 1)]
    mapping = {lbl: i for i, lbl in enumerate(top)}
    other = len(top)
    for lbl in counts:
        mapping.setdefault(lbl, other)
    return mapping
