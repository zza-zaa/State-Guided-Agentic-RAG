# Model Selection

## Recommended default stack

- Main LLM: `Qwen/Qwen3-14B`
- Fallback/debug LLM: `Qwen/Qwen2.5-14B-Instruct`
- High-accuracy optional LLM: `Qwen/Qwen3-32B`
- Embedding: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`
- Entity extraction: `urchade/gliner_medium-v2`

## Why this stack

### Qwen3-14B
Best balance for a single H100 80GB. It is agent-capable, supports thinking/non-thinking mode, and has a native 32K context with YaRN support for longer inputs.

### Qwen2.5-14B-Instruct
Very stable fallback for debugging, ablation, and compatibility checks.

### Qwen3-32B
Optional stronger model for final ablations and paper tables when more GPU memory or more GPUs are available.

### BGE-M3
Strong retrieval backbone for multi-granularity retrieval and longer chunks.

### BGE-reranker-v2-m3
Simple, strong, and easy to integrate with FlagEmbedding.

### GLiNER-medium-v2
Open-source entity extraction model with permissive Apache-2.0 license.
