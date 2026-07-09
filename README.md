# State-Guided Agentic RAG for Multi-Hop Question Answering

This repository implements a state-guided agentic Retrieval-Augmented Generation framework for multi-hop question answering.

The core idea is to formulate multi-hop QA as an iterative state resolution process. Instead of allowing an LLM agent to decide its next retrieval action only from implicit reasoning history, the system maintains an explicit, evidence-grounded knowledge state and uses it to guide retrieval, evidence completion, retry, stopping, and answer grounding.

## Motivation

Standard RAG usually follows a retrieve-then-read pipeline:

    question -> retrieve passages -> generate answer

This paradigm is often insufficient for multi-hop QA, where the answer depends on intermediate entities, relations, and evidence dependencies distributed across multiple documents.

Agentic RAG methods introduce iterative retrieval and reasoning actions, but their next actions are often decided from implicit LLM reasoning histories. This makes the agent vulnerable to unsupported action decisions and hallucination propagation.

Our framework argues that the key missing component in agentic RAG is not simply more actions, but an explicit evidence-grounded state that justifies and controls those actions.

## Method Overview

The framework maintains an explicit knowledge state during inference. The state contains:

- unresolved slots;
- dependency relations;
- candidate values;
- confirmed evidence;
- conflicts.

At each step, the state is updated with retrieved evidence and then used to guide the next action.

The overall pipeline is:

    Question
      -> State Initialization
      -> State-Guided Retrieval
      -> Evidence-Grounded State Update
      -> Retry / Second-Hop Retrieval / Dependency Query
      -> Answer Grounding
      -> Final Answer

## Main Features

- Explicit state modeling for multi-hop QA;
- State-guided retrieval control;
- Evidence-grounded state update;
- Dependency-aware query construction;
- Second-hop retrieval;
- Typed retry;
- Answer grounding;
- Evaluation on HotpotQA, 2Wiki, and MuSiQue;
- Ablation and step-budget analysis;
- Efficiency and token-cost profiling.

## Repository Structure

    .
    ├── configs/
    ├── scripts/
    ├── src/
    │   └── csa_rag/
    ├── README.md
    └── requirements.txt

Large files such as datasets, FAISS indexes, model checkpoints, experiment outputs, logs, and cache files are intentionally excluded from this open-source package.

## Installation

Create an environment and install dependencies:

    conda create -n state-rag python=3.10 -y
    conda activate state-rag
    pip install -r requirements.txt

You may need to modify paths in the configuration files according to your local environment.

## External Resources

This repository does not include datasets, indexes, or model weights.

Please prepare the following resources separately:

- HotpotQA
- 2WikiMultiHopQA
- MuSiQue
- Qwen3 model weights
- BGE embedding model
- BGE reranker
- Dense retrieval indexes

Example dataset paths:

    /path/to/dataset/hotpot/hotpot_dev_distractor_v1.json
    /path/to/dataset/2Wiki/dev.json
    /path/to/dataset/musique/musique_ans_v1.0_dev.jsonl

Example index paths:

    data/indexes/hotpot/
    data/indexes/2wiki/
    data/indexes/musique/

## Running Evaluation

HotpotQA example:

    export PYTHONPATH=src
    export CUDA_VISIBLE_DEVICES=0

    python -u scripts/run_hotpot_eval.py \
      --dataset-path /path/to/hotpot_dev_distractor_v1.json \
      --index-dir data/indexes/hotpot \
      --models-config configs/main_ablations/main_full_14b.yaml \
      --output-path outputs/hotpot_dev_predictions.jsonl \
      --limit 500 \
      --offset 0

2Wiki example:

    export PYTHONPATH=src
    export CUDA_VISIBLE_DEVICES=0

    python -u scripts/run_2wiki_eval.py \
      --dataset-path /path/to/2Wiki/dev.json \
      --index-dir data/indexes/2wiki \
      --models-config configs/main_ablations/main_full_14b.yaml \
      --output-path outputs/2wiki_dev_predictions.jsonl \
      --limit 500 \
      --offset 0

MuSiQue example:

    export PYTHONPATH=src
    export CUDA_VISIBLE_DEVICES=0

    python -u scripts/run_musique_eval.py \
      --dataset-path /path/to/musique_ans_v1.0_dev.jsonl \
      --index-dir data/indexes/musique \
      --models-config configs/main_ablations/main_full_14b.yaml \
      --output-path outputs/musique_dev_predictions.jsonl \
      --limit 500 \
      --offset 0

## Evaluation Metrics

The project reports:

- Exact Match;
- token-level F1;
- average state steps;
- error count;
- inference time;
- LLM call count;
- tokenizer-estimated token cost.

Metric computation example:

    python scripts/eval_em_f1.py \
      --predictions outputs/hotpot_dev_predictions.jsonl \
      --output outputs/hotpot_metrics.json

## Ablation Study

Main ablation variants include:

- main_state_only_14b;
- main_no_question_type_14b;
- main_no_state_optimization_14b;
- main_full_14b.

Example:

    bash scripts/run_main_ablation_limit.sh main_full_14b hotpot 500 0

## Step-Budget Analysis

Example:

    bash scripts/run_fixed_step_limit.sh 3 hotpot 100 0

This evaluates the model with a fixed state-step budget of 3 on 100 HotpotQA examples.

## Token Cost Profiling

Token consumption is estimated over all LLM calls. For each question, every prompt input and model-generated output is tokenized with the Qwen tokenizer and summed over the full inference process.

Example:

    python scripts/profile_ours_all_llm_tokens_500.py \
      --dataset-name hotpot \
      --dataset-path /path/to/hotpot_dev_distractor_v1.json \
      --index-dir data/indexes/hotpot \
      --models-config configs/main_ablations/main_full_14b.yaml \
      --tokenizer-path /path/to/Qwen3-14B \
      --limit 500 \
      --offset 0 \
      --output-summary outputs/llm_token_profile/hotpot_summary.json \
      --output-details outputs/llm_token_profile/hotpot_details.jsonl

## Notes

This repository is released as source code only. It does not include:

- raw datasets;
- processed datasets;
- FAISS indexes;
- model weights;
- reranker weights;
- experiment outputs;
- logs;
- checkpoints;
- cache files.

Please download datasets and models from their official sources and update configuration paths accordingly.

## Citation

    @misc{state_guided_agentic_rag,
      title = {State-Guided Agentic Retrieval-Augmented Generation for Multi-Hop Question Answering},
      author = {Anonymous},
      year = {2026},
      note = {Code release}
    }

## License

Please add an appropriate open-source license before public release, such as MIT, Apache-2.0, or another license approved by your institution.
