# State-Guided Agentic RAG for Multi-Hop Question Answering

This repository provides the implementation of State-Guided Agentic
Retrieval-Augmented Generation (State-Guided RAG) for multi-hop question
answering.

The framework formulates multi-hop QA as an iterative state resolution
process. Instead of relying only on implicit LLM reasoning history, it
maintains an explicit evidence-grounded state containing unresolved
slots, dependency relations, candidate values, confirmed evidence, and
conflicts. The state guides retrieval, verification, retry, stopping,
and answer grounding.

## Overview

Traditional RAG follows:

Question -\> Retrieve -\> Read -\> Answer

State-Guided RAG introduces:

Question -\> State Initialization -\> State-Guided Retrieval -\>
Evidence Collection -\> State Update -\> Verification/Retry -\> Answer
Grounding

## Core Components

The state contains:

-   unresolved slots
-   dependency relations
-   candidate values
-   confirmed evidence
-   conflicts

The state controls:

-   dependency-aware retrieval
-   second-hop retrieval
-   typed retry
-   conflict resolution
-   answer grounding

## Repository Structure

    configs/
    prompts/
    scripts/
    src/
    docs/
    examples/
    data/
    outputs/
    logs/

Large resources such as datasets, checkpoints, indexes, and experiment
outputs are excluded.

## Environment Setup

Recommended:

-   Python \>= 3.10
-   CUDA \>= 12.0
-   PyTorch \>= 2.3

Installation:

    conda create -n state-rag python=3.10 -y
    conda activate state-rag
    pip install -r requirements.txt

## Model Preparation

Required models:

LLM: Qwen3-14B (default experimental model)

Embedding: BAAI/bge-m3

Reranker: BAAI/bge-reranker-large

## Dataset Preparation

Supported datasets:

-   HotpotQA
-   2WikiMultiHopQA
-   MuSiQue

Data processing:

1.  Load raw QA files.
2.  Convert questions, answers, contexts, and supporting facts into
    unified format.
3.  Build retrieval corpus.
4.  Encode corpus with BGE-M3.
5.  Construct FAISS indexes.
6.  Store metadata mapping embeddings to documents.

Generated index files:

    data/indexes/
    ├── index.faiss
    ├── embeddings.npy
    └── metadata.jsonl

## Running Evaluation

Example:

    export PYTHONPATH=src

    python scripts/run_hotpot_eval.py  --dataset-path /path/to/data  --index-dir data/indexes/hotpot  --models-config configs/main_ablations/main_full_14b.yaml  --limit 500

Similar scripts are provided for 2Wiki and MuSiQue.

## Configuration

Generation uses deterministic decoding:

    temperature=0.0
    do_sample=false
    top_p=1.0

## Evaluation Metrics

The framework reports:

-   Exact Match
-   token-level F1
-   average state steps
-   inference latency
-   LLM calls
-   token consumption

## Ablation and Analysis

Supported studies include:

-   state-only ablation
-   question-type ablation
-   state optimization ablation
-   step-budget analysis
-   token-cost profiling

## Reproducibility

Steps:

1.  Install dependencies.
2.  Download datasets.
3.  Download model weights.
4.  Build retrieval indexes.
5.  Modify configuration paths.
6.  Run evaluation scripts.

## Notes

Not included:

-   datasets
-   model weights
-   FAISS indexes
-   generated outputs
-   logs
-   cache files

Users should download required resources separately.

## License

Please add an appropriate open-source license before public release,
such as MIT or Apache-2.0.
