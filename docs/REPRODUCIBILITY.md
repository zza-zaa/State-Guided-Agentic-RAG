# Reproducibility Notes

To reproduce the reported experiments, keep the following fixed:

1. Dataset versions and development splits.
2. Corpus extraction scripts.
3. Chunking parameters.
4. BGE-M3 embedding model and normalization setting.
5. Reranker model and reranking top-k.
6. LLM backend and decoding settings.
7. Random seed.
8. Answer normalization and EM/F1 script.
9. Evaluation subsets selected by fixed `limit` and `offset`.

The original release uses source code only. Large files such as model weights, raw datasets, processed corpora, FAISS indexes, outputs, and logs must be prepared locally.
