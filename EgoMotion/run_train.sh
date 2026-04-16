#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for full training only.
# Usage:
#   bash run_train.sh
# Optional env vars:
#   PYTHON_BIN=python
#   MANIFEST=adt_pipeline_data/processed/manifest.json
#   PRECOMPUTE_OUT_DIR=outputs/precompute_cache

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
MANIFEST="${MANIFEST:-adt_pipeline_data/processed/manifest.json}"
PRECOMPUTE_OUT_DIR="${PRECOMPUTE_OUT_DIR:-outputs/precompute_cache}"
MODEL_CONFIG="configs/model.yaml"
FULL_DATA_CONFIG="configs/data.yaml"
FULL_TRAIN_CONFIG="configs/train.yaml"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs logs checkpoints

echo "[run_train] mode=full"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" tools/train.py \
  --manifest "$MANIFEST" \
  --data-config "$FULL_DATA_CONFIG" \
  --model-config "$MODEL_CONFIG" \
  --train-config "$FULL_TRAIN_CONFIG" \
  --split-dir outputs/splits \
  --precompute-before-train \
  --precompute-out-dir "$PRECOMPUTE_OUT_DIR"
