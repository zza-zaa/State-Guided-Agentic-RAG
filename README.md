# State-Guided-Agentic-Retrieval-Augmented-Generation-for-Multi-Hop-Question-Answering


This repository contains the source code for **State-Guided-Agentic-Retrieval-Augmented-Generation-for-Multi-Hop-Question-Answering**, a framework for multi-hop question answering that treats agentic retrieval as an explicit state resolution process.

Instead of relying only on implicit LLM reasoning traces, the system maintains an evidence-grounded structured state during inference. The state records unresolved information needs, dependency relations, candidate values, confirmed evidence, and conflicts. It is updated with retrieved evidence and then used to guide retrieval, evidence completion, retry, stopping, and answer grounding.

## Overview

Standard retrieval-augmented generation usually follows a retrieve-then-read pipeline:

```text
question -> retrieve passages -> generate answer
```

This is often insufficient for multi-hop question answering, where an answer depends on intermediate entities, relations, and evidence dependencies distributed across multiple documents.

Agentic RAG methods introduce iterative retrieval and reasoning actions, but many of them decide the next action from free-form reasoning histories or generated subqueries. This makes the retrieval process difficult to inspect and may propagate unsupported intermediate predictions.

This project introduces an explicit **evidence-grounded state** as the action-grounding interface between retrieval and generation.

At a high level, the pipeline is:

```text
Question
  -> Typed State Induction
  -> State-Guided Retrieval
  -> Evidence-Grounded State Update
  -> Dependency-Aware Retrieval / Second-Hop Retrieval / Typed Retry
  -> State-Grounded Answer Generation
  -> Final Answer
```

## Main Features

- Explicit state modeling for multi-hop QA.
- Typed state induction with slots, dependencies, and answer-target roles.
- Evidence-grounded state update with missing, candidate, confirmed, and conflict statuses.
- Dependency-aware query construction.
- Second-hop retrieval and title-hop retrieval.
- Typed retry for unresolved slots.
- State-grounded answer selection and answer rewriting.
- Evaluation scripts for HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- Main ablation, macro ablation, fixed-step, efficiency, token-cost, and error-analysis scripts.
- Support for local LLM backends and OpenAI-compatible API backends.

## Repository Structure

```text
.
├── configs/
│   ├── models.yaml
│   ├── models_qwen3_4b.yaml
│   ├── models_qwen3_8b.yaml
│   ├── pipeline.yaml
│   ├── main_ablations/
│   ├── ablations/
│   ├── macro_ablations/
│   └── fixed_steps/
│
├── prompts/
│   ├── decompose.txt
│   ├── state_update.txt
│   ├── answer.txt
│   ├── select_answer.txt
│   └── rewrite_answer.txt
│
├── src/csa_rag/
│   ├── agent/
│   │   ├── pipeline.py
│   │   ├── router.py
│   │   └── calibration.py
│   ├── embedding/
│   │   └── bge_embedder.py
│   ├── llm/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── transformers_llm.py
│   │   ├── vllm_llm.py
│   │   └── openai_compat_llm.py
│   ├── rerank/
│   │   └── bge_reranker.py
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   ├── typed_second_hop.py
│   │   ├── typed_retry_builder.py
│   │   └── slot_router.py
│   ├── state/
│   │   ├── decomposer.py
│   │   ├── tracker.py
│   │   ├── question_typing.py
│   │   ├── question_typing_v3.py
│   │   ├── question_typing_v4.py
│   │   └── state_critic.py
│   ├── tools/
│   │   └── entity_extractor.py
│   ├── utils/
│   │   ├── answer_utils.py
│   │   ├── direct_answer.py
│   │   ├── typed_answer_grounder.py
│   │   ├── typed_target_selector.py
│   │   ├── json_utils.py
│   │   └── support_path.py
│   ├── config.py
│   └── schema.py
│
├── scripts/
│   ├── prepare_hotpot_corpus.py
│   ├── prepare_2wiki_corpus.py
│   ├── prepare_musique_corpus.py
│   ├── build_index.py
│   ├── run_hotpot_eval.py
│   ├── run_2wiki_eval.py
│   ├── run_musique_eval.py
│   ├── eval_em_f1.py
│   ├── run_main_ablation_limit.sh
│   ├── run_fixed_step_limit.sh
│   ├── profile_ours_all_llm_tokens_500.py
│   └── ...
│
├── requirements.txt
├── environment.yml
├── MODEL_SELECTION.md
├── OFFLINE_DEPLOYMENT.md
└── README.md
```

Large files are intentionally excluded from this repository, including raw datasets, processed corpora, FAISS indexes, model weights, reranker weights, experiment outputs, logs, and caches.

## Installation

We recommend Python 3.10.

```bash
conda create -n state-rag python=3.10 -y
conda activate state-rag

pip install -r requirements.txt
```

The core dependencies include:

```text
torch
transformers
accelerate
sentence-transformers
faiss-cpu
numpy
scipy
pandas
tqdm
pyyaml
jsonlines
scikit-learn
openai
httpx
vllm
FlagEmbedding
rank-bm25
regex
```

If you use GPU FAISS, install a FAISS package compatible with your CUDA and PyTorch versions. If GPU FAISS causes compatibility issues, `faiss-cpu` is sufficient for reproducing the pipeline at moderate scale.

## Required External Resources

This repository does not include datasets, model weights, reranker weights, or FAISS indexes. Please prepare them separately.

### Models

The default configuration assumes the following resources:

```text
/path/to/models/Qwen3-14B
/path/to/models/Qwen3-8B
/path/to/models/Qwen3-4B
/path/to/models/bge-m3
/path/to/models/bge-reranker-v2-m3
```

You may also use other local LLMs such as Llama or Mistral by modifying the `llm.default_model` field in the configuration file.

Example:

```yaml
llm:
  backend: vllm
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
```

### OpenAI-Compatible API Backend

The repository also contains an OpenAI-compatible backend in:

```text
src/csa_rag/llm/openai_compat_llm.py
```

To use an OpenAI-compatible server or API, set the backend to `openai_compat`:

```yaml
runtime:
  use_vllm: false
  llm_backend: openai_compat

llm:
  backend: openai_compat
  server_base_url: https://api.openai.com/v1
  server_api_key: ${OPENAI_API_KEY}
  server_model_name: gpt-4o-mini
  max_new_tokens: 512
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  thinking: false
```

Then export your API key before running:

```bash
export OPENAI_API_KEY=your_api_key
```

Depending on your API provider, you may need to adjust the model name and endpoint.

## Data Preparation

The code supports HotpotQA, 2WikiMultiHopQA, and MuSiQue. The repository does not redistribute these datasets. Please download them from their official sources and place them under a local data directory.

A recommended layout is:

```text
data/
├── raw/
│   ├── hotpot/
│   │   └── hotpot_dev_distractor_v1.json
│   ├── 2wiki/
│   │   └── dev.json
│   └── musique/
│       └── musique_ans_v1.0_dev.jsonl
│
├── corpus/
│   ├── hotpot_corpus.jsonl
│   ├── 2wiki_corpus.jsonl
│   └── musique_corpus.jsonl
│
└── indexes/
    ├── hotpot/
    ├── 2wiki/
    └── musique/
```

The evaluation scripts expect question-answer files, while the retrieval module expects a corpus JSONL file and a FAISS index.

Each corpus file should contain one document per line:

```json
{"id": "doc_id", "title": "document title", "text": "document text"}
```

## Building Corpora

### HotpotQA

```bash
python scripts/prepare_hotpot_corpus.py \
  --input-dir data/raw/hotpot \
  --output data/corpus/hotpot_corpus.jsonl
```

### 2WikiMultiHopQA

```bash
python scripts/prepare_2wiki_corpus.py \
  --input-dir data/raw/2wiki \
  --output data/corpus/2wiki_corpus.jsonl
```

### MuSiQue

```bash
python scripts/prepare_musique_corpus.py \
  --input-dir data/raw/musique \
  --output data/corpus/musique_corpus.jsonl
```

These scripts scan JSON/JSONL files under the input directory, extract document titles and text fields, deduplicate documents with MD5 hashes, and write a normalized JSONL corpus.

## Building FAISS Indexes

After preparing the corpus files, build indexes with BGE-M3 embeddings:

```bash
python scripts/build_index.py \
  --input data/corpus/hotpot_corpus.jsonl \
  --output data/indexes/hotpot \
  --model-name /path/to/models/bge-m3
```

```bash
python scripts/build_index.py \
  --input data/corpus/2wiki_corpus.jsonl \
  --output data/indexes/2wiki \
  --model-name /path/to/models/bge-m3
```

```bash
python scripts/build_index.py \
  --input data/corpus/musique_corpus.jsonl \
  --output data/indexes/musique \
  --model-name /path/to/models/bge-m3
```

The default index-building script uses character-level chunking. In `scripts/build_index.py`, the chunker is initialized as:

```python
Chunker(min_chars=300, max_chars=1200, overlap=120)
```

The main experiment configuration may use a different chunk setting, for example:

```yaml
retrieval:
  min_chunk_chars: 150
  max_chunk_chars: 600
  chunk_overlap: 80
```

Please keep the chunking configuration consistent when reproducing reported experiments.

## Configuration

The most important configuration file is:

```text
configs/main_ablations/main_full_14b.yaml
```

Important fields include:

```yaml
runtime:
  device: cuda
  dtype: bfloat16
  seed: 42
  use_vllm: true
  llm_backend: vllm

llm:
  backend: vllm
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
  slot_confidence_floor: 0.25
  max_evidence_per_step: 3

pipeline:
  enable_typed_question_spec: true
  enable_second_hop: true
  enable_dependency_query: true
  enable_title_hop: true
  enable_typed_retry: true
  enable_typed_answer_grounding: true
  enable_typed_target_selector: false
```

## Running Evaluation

Set the project path:

```bash
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0
```

### HotpotQA

```bash
python -u scripts/run_hotpot_eval.py \
  --dataset-path data/raw/hotpot/hotpot_dev_distractor_v1.json \
  --index-dir data/indexes/hotpot \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/hotpot_dev_predictions.jsonl \
  --limit 500 \
  --offset 0
```

### 2WikiMultiHopQA

```bash
python -u scripts/run_2wiki_eval.py \
  --dataset-path data/raw/2wiki/dev.json \
  --index-dir data/indexes/2wiki \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/2wiki_dev_predictions.jsonl \
  --limit 500 \
  --offset 0
```

### MuSiQue

```bash
python -u scripts/run_musique_eval.py \
  --dataset-path data/raw/musique/musique_ans_v1.0_dev.jsonl \
  --index-dir data/indexes/musique \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/musique_dev_predictions.jsonl \
  --limit 500 \
  --offset 0
```

Each prediction file is written as JSONL. Each row contains:

```json
{
  "qid": "...",
  "question": "...",
  "gold_answer": "...",
  "pred_answer": "...",
  "raw_pred_answer": "...",
  "steps": 3,
  "state": {},
  "retrieved_evidence": [],
  "rationale": "...",
  "error": null
}
```

## Computing EM and F1

```bash
python scripts/eval_em_f1.py \
  --predictions outputs/hotpot_dev_predictions.jsonl \
  --output outputs/hotpot_metrics.json
```

The script reports:

```text
total
error_count
EM
F1
avg_steps
```

## Ablation Experiments

Main ablation configurations are in:

```text
configs/main_ablations/
```

The included variants are:

```text
main_full_14b.yaml
main_state_only_14b.yaml
main_no_question_type_14b.yaml
main_no_state_optimization_14b.yaml
```

Example:

```bash
bash scripts/run_main_ablation_limit.sh main_full_14b hotpot 500 0
```

Patch-level ablation configurations are in:

```text
configs/ablations/
```

Macro-stage ablations are in:

```text
configs/macro_ablations/
```

Fixed-step configurations are in:

```text
configs/fixed_steps/
```

## Fixed State-Step Analysis

Example:

```bash
bash scripts/run_fixed_step_limit.sh 3 hotpot 100 0
```

This evaluates the model with a fixed state-step budget of 3 on 100 HotpotQA examples.

## Efficiency and Token Profiling

The repository includes scripts for runtime and token-cost profiling.

Example:

```bash
python scripts/profile_ours_all_llm_tokens_500.py \
  --dataset-name hotpot \
  --dataset-path data/raw/hotpot/hotpot_dev_distractor_v1.json \
  --index-dir data/indexes/hotpot \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --tokenizer-path /path/to/models/Qwen3-14B \
  --limit 500 \
  --offset 0 \
  --output-summary outputs/llm_token_profile/hotpot_summary.json \
  --output-details outputs/llm_token_profile/hotpot_details.jsonl
```

Token consumption is estimated over all LLM calls. For each question, every prompt input and model-generated output is tokenized with the selected tokenizer and summed over the complete inference process.

## Error Analysis

The repository contains several utilities for error analysis:

```bash
python scripts/classify_errors.py
python scripts/extract_errors.py
python scripts/sample_errors_by_type.py
python scripts/emnlp_error_audit.py
```

These scripts can be adapted to inspect failed cases, compare error types, and sample predictions for qualitative analysis.

## Running a Quick Smoke Test

After preparing a small dataset and index, run:

```bash
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

python -u scripts/run_hotpot_eval.py \
  --dataset-path data/raw/hotpot/hotpot_dev_distractor_v1.json \
  --index-dir data/indexes/hotpot \
  --models-config configs/main_ablations/main_full_14b.yaml \
  --output-path outputs/smoke_hotpot.jsonl \
  --limit 5 \
  --offset 0
```

Then compute metrics:

```bash
python scripts/eval_em_f1.py \
  --predictions outputs/smoke_hotpot.jsonl \
  --output outputs/smoke_hotpot_metrics.json
```

## Notes on Reproducibility

The original experiments used:

- BGE-M3 as the dense embedding model.
- A BGE reranker as the cross-encoder reranker.
- Qwen3 family models as LLM backbones.
- Character-level chunking.
- Fixed evaluation subsets selected by `limit` and `offset`.
- Random seed 42.

For exact reproduction, please ensure that:

1. Dataset versions match the ones used in the paper.
2. Corpus extraction and chunking settings are unchanged.
3. Dense indexes are rebuilt with the same embedding model.
4. The same LLM backend and decoding settings are used.
5. The same answer normalization and `eval_em_f1.py` script are used.

## What Is Not Included

This source-code release does not include:

```text
raw datasets
processed datasets
FAISS indexes
model weights
embedding weights
reranker weights
experiment outputs
logs
checkpoints
cache files
private server paths
```

Please download datasets and models from their official sources and update paths in the configuration files.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@misc{state_guided_agentic_rag_2026,
  title        = {State-Guided Agentic Retrieval-Augmented Generation for Multi-Hop Question Answering},
  author       = {Anonymous},
  year         = {2026},
  note         = {Code release}
}
```

Please replace this placeholder citation with the official citation after publication.

## License

Please add an open-source license before public release. Common choices include:

- MIT License
- Apache License 2.0
- BSD 3-Clause License

Check your institution's policy before releasing the code.
