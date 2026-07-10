#!/usr/bin/env bash
set -e

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

python -u scripts/run_musique_eval.py \
  --dataset-path data/raw/musique/musique_ans_v1.0_dev.jsonl \
  --index-dir data/indexes/musique \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/smoke_musique.jsonl \
  --limit 5 \
  --offset 0

python scripts/eval_em_f1.py \
  --predictions outputs/smoke_musique.jsonl \
  --output outputs/smoke_musique_metrics.json
