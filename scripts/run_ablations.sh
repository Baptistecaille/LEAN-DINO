#!/usr/bin/env bash
# The experiment the project exists to run. Same compute budget for all three arms.
# certified - dropout_only is the result.
set -euo pipefail
CFG=${1:-configs/dino_v0.yaml}
for MODE in dropout_only naive certified; do
  echo "=== $MODE ==="
  python scripts/train_dino.py --config "$CFG" --view-mode "$MODE"
  python scripts/eval_all.py --config "$CFG" \
    --ckpt "outputs/dino_v0_certified_${MODE}/last.pt" || true
done
