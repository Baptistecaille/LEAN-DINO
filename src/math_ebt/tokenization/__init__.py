"""Tokenizer loading, and the one place where the model is sized to it."""
from __future__ import annotations

from tokenizers import Tokenizer

from math_ebt.config import Config


def load_tokenizer(cfg: Config) -> Tokenizer:
    """Load the trained tokenizer and resize `cfg.model.vocab_size` to match it.

    `model.vocab_size` in the YAML is the BPE *training target*; the saved
    tokenizer is always larger, because `train_tokenizer.py` calls `add_tokens`
    afterwards to make frequent Mathlib constants atomic (32000 -> 33464 on the
    full corpus). Sizing `nn.Embedding` from the YAML value therefore leaves the
    top ~1.5k token ids with no row: on CUDA that is a device-side assert
    (`vectorized_gather_kernel index out of bounds`), on MPS/CPU it is silently
    wrong, which is worse.

    Every entrypoint that builds a model must obtain its tokenizer through this
    function. Rejected alternative: training BPE to `vocab_size - top_constants`
    so the total lands on the YAML number -- `add_tokens` deduplicates against
    the learned vocabulary, so the total stays unpredictable and the invariant
    would still rest on a coincidence rather than on the tokenizer being the
    single source of truth.
    """
    tok = Tokenizer.from_file(cfg.data.tokenizer_path)
    cfg.model.vocab_size = tok.get_vocab_size()
    return tok
