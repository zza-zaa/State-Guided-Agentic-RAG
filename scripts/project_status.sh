#!/usr/bin/env bash
set -euo pipefail

cd /mnt/raid/peiyu/csa

echo "==== CONFIG ===="
sed -n '1,120p' configs/models.yaml

echo
echo "==== CORPORA ===="
ls -lh data/corpus || true

echo
echo "==== INDEXES ===="
find data/indexes -maxdepth 2 -type f | sort || true

echo
echo "==== HOTPOT MAIN ===="
[ -f outputs/hotpot_dev100_metrics.json ] && cat outputs/hotpot_dev100_metrics.json || echo "missing"

echo
echo "==== HOTPOT BASELINES ===="
[ -f outputs/hotpot_dev100_baseline_dense_metrics.json ] && cat outputs/hotpot_dev100_baseline_dense_metrics.json || echo "dense missing"
[ -f outputs/hotpot_dev100_baseline_rerank_metrics.json ] && cat outputs/hotpot_dev100_baseline_rerank_metrics.json || echo "rerank missing"
