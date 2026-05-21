#!/usr/bin/env bash
set -euo pipefail
cd F:/dataset/LUT_AD_DataSet

# Step 1: precompute CLIP tokens (idempotent — skips samples already cached)
python -m scripts.pointad_plus.precompute_clip_tokens \
  --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
  --out external/datasets/welds_pointad_mp/clip_tokens/

# Step 2: train fusion block on the cached tokens via SSL modality-masking
mkdir -p results/welds_pointad_plus
python -m src.pointad_plus.ssl_modality_masking \
  --tokens external/datasets/welds_pointad_mp/clip_tokens/ \
  --out results/welds_pointad_plus/fusion_block.pt \
  --epochs 20 --batch_size 16
