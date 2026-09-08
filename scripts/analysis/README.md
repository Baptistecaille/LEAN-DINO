# `scripts/analysis/` — post-hoc analysis of an existing checkpoint

No training. `cache_embeddings.py`, `probe_diagnostic.py`, `spectrum.py`, and
`view_triviality.py` run one forward pass over *sampled* splits, then everything
else runs on cached embeddings — cheap enough for a laptop (MPS or CPU).
`retrieval_centering.py` is the exception: it encodes the *full* retrieval corpus
(~307k declarations, not a sample), because `top10_jaccard` and dense recall are
defined against it. Budget a real GPU pass for that one (see its own docstring).

They exist to answer four questions the phase-1 gate table cannot:

1. Is `probe_top1 = 0.538` a property of the encoder, or of how the probe was sampled?
2. What does the latent geometry look like, given `cos_pos = 0.98` with `top10_jaccard = 0.26`?
3. Could an untrained encoder pass the `cos_pos` gate?
4. Given (2), does removing the corpus-wide shared direction before ranking recover
   `top10_jaccard` — and does it close any of the dense-vs-BM25 retrieval gap?

## The finding that motivated all of this

`scripts/eval_all.py` slices the probe sets by prefix:

```python
probe_train = train[: args.max_probe_train]   # train[:20000]
probe_valid = valid[: args.max_probe_valid]   # valid[:5000]
```

The splits are written in corpus order — `schema.read_dir` does
`sorted(Path(raw_dir).glob("*.jsonl"))` — so a prefix is an **alphabetical prefix of
Mathlib**. Counted on `data/splits`:

| slice | distinct domains | majority share |
|---|---|---|
| `train[:20000]` | **1** (`Algebra`) | 100.00 % |
| `valid[:5000]` | 5 | **53.86 %** |
| full `valid` | 27 | 17.8 % (`Algebra`) |

53.86 % is exactly the `probe_majority_baseline` in `eval.json`, and `probe_top1 =
0.5380` is what a probe trained on a single class scores against it. The reported
figure measures the slicing, not the representation. `macro_f1 = 0.100` follows
mechanically.

`query_pool` is unaffected — it already uses `rng.sample(test, ...)`. The bug is
confined to the two probe slices.

### The fix in `eval_all.py`

```diff
-    probe_train = train[: args.max_probe_train]
-    probe_valid = valid[: args.max_probe_valid]
+    # Prefix slicing is not a subsample here: the splits are written in corpus
+    # order, so train[:20000] is 100% Algebra on this corpus and the probe trains
+    # on a single class. A separate Random keeps `rng` untouched, so query_pool
+    # stays byte-identical to previous runs.
+    probe_rng = random.Random(args.seed)
+    probe_train = probe_rng.sample(train, min(args.max_probe_train, len(train)))
+    probe_valid = probe_rng.sample(valid, min(args.max_probe_valid, len(valid)))
```

Deterministic under `--seed`, so it stays reusable across ablation arms. The label
map is already built from the full train split and needs no change.

## Running them

Order matters: `cache_embeddings.py` writes the `.npz` the others read.

```bash
# 1. encode sampled splits once  (the only step that touches the model; ~10-30 min)
python scripts/analysis/cache_embeddings.py \
    --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt

# 2. probe, corrected sampling vs the prefix slicing  (adds ~25k more encodes)
python scripts/analysis/probe_diagnostic.py \
    --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt
#    --skip-prefix  to run only the corrected arm

# 3. latent geometry  (seconds, cached embeddings only)
python scripts/analysis/spectrum.py --split test

# 4. is the cos_pos gate falsifiable?  (~3 min: 2000 pairs x 2 encoders)
python scripts/analysis/view_triviality.py \
    --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt

# 5. does centering fix retrieval?  (tens of minutes: encodes the FULL 307k corpus)
python scripts/analysis/retrieval_centering.py \
    --config configs/dino_v0.yaml --ckpt model/model_20000_step.pt
#    --whiten-dims 32   to also try full PCA whitening, not just mean removal
```

Everything lands in `outputs/analysis/`. Pass the same `--seed` across ablation
arms, for the reason `eval_all.py` fixes its own: sampling noise otherwise leaks
into the C − A comparison that is the actual result.

## Reading the output

**`probe_diagnostic.py`** — the `sampled` row is the number worth quoting; the
`prefix` row is the evidence that the published one is an artefact. `pred cls` is
how many distinct classes the probe ever predicts: 1 means it degenerated to the
majority class regardless of input. `balAcc` (mean per-class recall) is the metric
to prefer over `top1` on a distribution this skewed — `top1` on a 53.9 % majority is
not interpretable, which is also why the `probe_top1 ≥ 0.55` gate is badly placed:
it sits 1.1 points above the baseline of a slice that should not have been used.

**`spectrum.py`** — `effective_rank` against `d_model = 384` says how much of the
space the encoder actually uses. `mean_norm_ratio` near 1 means a single shared
component dominates every vector, in which case a high `cos_pos` is mostly measuring
that component. The `cos_rand` curve separates the mean from the leading directions:
a sharp drop means the anisotropy is concentrated in a few axes and post-hoc
whitening is worth trying before spending GPU hours; a flat curve means it is not.

**`view_triviality.py`** — the table is four conditions and only the comparisons
matter. If *untrained, matched* already clears 0.80, the gate does not test
training. If *trained, matched* ≈ *trained, mismatched*, `cos_pos` is measuring
anisotropy rather than invariance. Measured beforehand on `data/splits/test.jsonl`:
`type_alpha` is present on 100 % of declarations and only 2.7 % of certified views
are byte-identical to their anchor, with a mean whitespace-token Jaccard of 0.34 —
so the views are lexically real, and a high `cos_pos` is not explained by the
fallback chain in `eval_all.py` collapsing to the anchor's own text.

**`retrieval_centering.py`** — the `raw` row is a reproduction check: it should
land close to the published `top10_jaccard = 0.256` and `dense_recall@10 =
0.0128`. If it doesn't, don't trust the `centered`/`whitened` rows either —
something about corpus construction or query sampling has drifted. The comparison
that matters is `centered` vs `raw`: a large jump means the shared direction
`spectrum.py` found was masking real declaration-to-declaration invariance and a
cheap post-hoc fix (subtract the corpus mean before ranking) is worth adopting by
default; little to no change means the residual signal itself is too weak to
support fine-grained retrieval, and the fix has to come from more training or a
different objective, not from this kind of post-processing. Same reading applies
to `centered recall@10` against the printed BM25 reference.

## Design notes

- **Reservoir sampling, not `random.sample(list(read_jsonl(path)), n)`.** The
  obvious fix materialises the whole split first: `train.jsonl` is 2.2 GB on disk
  and ~3.9 GB as `Declaration` objects, per the note in `configs/dino_v0.yaml`.
  Algorithm R over raw JSON strings holds `n` lines and nothing else.
- **`common.build_label_map_streaming` is verified equal** to
  `data.dataset.build_label_map` on the same split; it exists only so the map can be
  built over full train without holding it in RAM.
- **`eval/probe.py` is not modified.** It is on the phase-1 result path and its
  numbers must stay comparable across ablation arms, so `probe_diagnostic.py` calls
  it for the headline metrics and refits its own copy for the predictions it needs.
- **No new dependencies** (torch, numpy, tokenizers, pyyaml, tqdm). Spearman is
  implemented on numpy ranks rather than pulling in scipy.
- **`retrieval_centering.py` reproduces `eval_all.py`'s invariance-slicing quirk on
  purpose.** `za`/`zv` are sliced to `len(queries)` by count, not re-filtered to
  the same declarations as `queries` (the premise-selection subset) — matching
  `eval_all.py` exactly so the `raw` row is a fair reproduction check. Worth fixing
  in `eval_all.py` itself at some point; not silently fixed here, since that would
  make the reproduction check meaningless.
- **Ranking tensors are moved to the accelerator explicitly.** `encode_texts`
  always returns CPU tensors (so batches don't pin GPU memory across the whole
  corpus); `retrieval_centering.py` re-uploads the full corpus once after encoding
  so the repeated raw/centered/whitened rankings run on the GPU instead of the CPU
  eval_all.py's own `dense_rank` implicitly runs on — a speed change only, same
  numbers.
