# Data Preparation

This repository does not include raw datasets, processed corpora, FAISS indexes, model weights, or experiment outputs.

Recommended local layout:

```text
data/
├── raw/
│   ├── hotpot/
│   ├── 2wiki/
│   └── musique/
├── corpus/
└── indexes/
```

## HotpotQA

Place the development file under:

```text
data/raw/hotpot/hotpot_dev_distractor_v1.json
```

Build the corpus:

```bash
python scripts/prepare_hotpot_corpus.py \
  --input-dir data/raw/hotpot \
  --output data/corpus/hotpot_corpus.jsonl
```

## 2WikiMultiHopQA

Place the 2Wiki files under:

```text
data/raw/2wiki/
```

Build the corpus:

```bash
python scripts/prepare_2wiki_corpus.py \
  --input-dir data/raw/2wiki \
  --output data/corpus/2wiki_corpus.jsonl
```

## MuSiQue

Place the MuSiQue files under:

```text
data/raw/musique/
```

Build the corpus:

```bash
python scripts/prepare_musique_corpus.py \
  --input-dir data/raw/musique \
  --output data/corpus/musique_corpus.jsonl
```

## Build Indexes

```bash
python scripts/build_index.py \
  --input data/corpus/hotpot_corpus.jsonl \
  --output data/indexes/hotpot \
  --model-name /path/to/models/bge-m3
```

Repeat the same command for 2Wiki and MuSiQue by changing the input corpus and output index directory.
