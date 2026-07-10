from __future__ import annotations

from typing import Iterable, List
import numpy as np
from FlagEmbedding import BGEM3FlagModel


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", normalize: bool = True, use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.normalize = normalize

    def encode(self, texts: Iterable[str], batch_size: int = 16, max_length: int = 2048) -> np.ndarray:
        texts = list(texts)
        outputs = self.model.encode(texts, batch_size=batch_size, max_length=max_length)
        vecs = outputs["dense_vecs"]
        vecs = np.asarray(vecs, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs
