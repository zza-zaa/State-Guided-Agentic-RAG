from __future__ import annotations
import json
import os
from pathlib import Path

def check_path(name: str, path: str):
    p = Path(path)
    print(f"[{name}] {path} -> {'OK' if p.exists() else 'MISSING'}")

def main():
    print("=== ENV ===")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
    print("TRANSFORMERS_OFFLINE =", os.environ.get("TRANSFORMERS_OFFLINE"))
    print("HF_DATASETS_OFFLINE =", os.environ.get("HF_DATASETS_OFFLINE"))

    print("\n=== MODELS ===")
    check_path("Qwen3-14B", "/mnt/raid/zsb/llm_models/Qwen3-14B")
    check_path("Qwen3-32B", "/mnt/raid/zsb/llm_models/Qwen3-32B")
    check_path("Qwen2.5-14B-Instruct", "/mnt/raid/peiyu/models/Qwen2.5-14B-Instruct")
    check_path("bge-m3", "/mnt/raid/peiyu/models/bge-m3")
    check_path("bge-reranker-v2-m3", "/mnt/raid/peiyu/models/bge-reranker-v2-m3")
    check_path("gliner_medium-v2.1", "/mnt/raid/peiyu/models/gliner_medium-v2.1")
    check_path("deberta-v3-base", "/mnt/raid/peiyu/models/deberta-v3-base")

    print("\n=== DATASETS ===")
    check_path("hotpot", "/mnt/raid/peiyu/dataset/hotpot")
    check_path("2Wiki", "/mnt/raid/peiyu/dataset/2Wiki")
    check_path("musique", "/mnt/raid/peiyu/dataset/musique")

    print("\n=== PROJECT ARTIFACTS ===")
    check_path("hotpot_corpus", "data/corpus/hotpot_corpus.jsonl")
    check_path("hotpot_index", "data/indexes/hotpot/index.faiss")
    check_path("hotpot_chunks", "data/indexes/hotpot/chunks.jsonl")

    print("\n=== PYTHON LIBS ===")
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    except Exception as e:
        print("torch ERROR:", e)

    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers ERROR:", e)

    try:
        import faiss
        print("faiss: OK")
    except Exception as e:
        print("faiss ERROR:", e)

    try:
        import vllm
        print("vllm:", vllm.__version__)
    except Exception as e:
        print("vllm ERROR:", e)

    try:
        import FlagEmbedding
        print("FlagEmbedding: OK")
    except Exception as e:
        print("FlagEmbedding ERROR:", e)

    try:
        import gliner
        print("gliner: OK")
    except Exception as e:
        print("gliner ERROR:", e)

if __name__ == "__main__":
    main()
