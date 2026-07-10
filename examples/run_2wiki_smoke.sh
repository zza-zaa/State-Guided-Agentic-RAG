#!/usr/bin/env bash
set -e

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

python -u scripts/run_2wiki_eval.py \
  --dataset-path data/raw/2wiki/dev.json \
  --index-dir data/indexes/2wiki \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/smoke_2wiki.jsonl \
  --limit 5 \
  --offset 0

python scripts/eval_em_f1.py \
  --predictions outputs/smoke_2wiki.jsonl \
  --output outputs/smoke_2wiki_metrics.json
