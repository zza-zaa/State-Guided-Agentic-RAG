#!/usr/bin/env bash
set -e

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

python -u scripts/run_hotpot_eval.py \
  --dataset-path data/raw/hotpot/hotpot_dev_distractor_v1.json \
  --index-dir data/indexes/hotpot \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/smoke_hotpot.jsonl \
  --limit 5 \
  --offset 0

python scripts/eval_em_f1.py \
  --predictions outputs/smoke_hotpot.jsonl \
  --output outputs/smoke_hotpot_metrics.json
