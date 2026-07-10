from __future__ import annotations

from typing import List
from csa_rag.schema import Evidence


class BGEReranker:
    def __init__(self, model_name: str = "", use_fp16: bool = True):
        self.model = None
        if model_name:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, evidence: List[Evidence], top_k: int = 10) -> List[Evidence]:
        if self.model is None:
            return evidence[:top_k]

        pairs = [[query, e.text] for e in evidence]
        scores = self.model.compute_score(pairs, normalize=True)
        ranked = sorted(zip(evidence, scores), key=lambda x: x[1], reverse=True)

        output = []
        for e, s in ranked[:top_k]:
            payload = e.model_dump()
            payload["score"] = float(s)
            output.append(Evidence(**payload))
        return output
