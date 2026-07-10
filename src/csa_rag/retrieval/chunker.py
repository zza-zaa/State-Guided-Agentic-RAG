from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunker:
    min_chars: int = 300
    max_chars: int = 1200
    overlap: int = 120

    def split(self, text: str) -> List[str]:
        text = " ".join(text.split())
        if len(text) <= self.max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = max(0, end - self.overlap)
        return chunks
