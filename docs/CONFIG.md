# Configuration Guide

The main configurations are stored in `configs/`.

Important fields:

```yaml
runtime:
  device: cuda
  dtype: bfloat16
  seed: 42
  use_vllm: true
  llm_backend: vllm

llm:
  default_model: /path/to/models/Qwen3-14B
  max_model_len: 4096
  max_new_tokens: 512
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  thinking: false

embedding:
  model_name: /path/to/models/bge-m3
  normalize: true
  max_length: 2048
  batch_size: 16

reranker:
  model_name: /path/to/models/bge-reranker-v2-m3
  use_fp16: true
  top_k: 10

retrieval:
  top_k_dense: 12
  top_k_rerank: 5
  min_chunk_chars: 150
  max_chunk_chars: 600
  chunk_overlap: 80

agent:
  max_steps: 4
  stop_confidence: 0.83
  stop_if_no_missing_slots: true
  max_evidence_per_step: 3
```

Replace all `/path/to/...` values with your local dataset, model, and index paths.
