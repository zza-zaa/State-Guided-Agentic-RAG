from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate_json(self, prompt: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError
