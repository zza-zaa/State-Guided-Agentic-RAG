from __future__ import annotations

import json
import re
from typing import Any
from urllib import request, error

from csa_rag.llm.base import BaseLLM
from csa_rag.utils.json_utils import loads_maybe_json


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 1024,
        thinking: bool = False,
        timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.timeout = timeout

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} calling {path}: {detail}") from e
        except error.URLError as e:
            raise RuntimeError(f"Failed to reach server at {self.base_url}{path}: {e}") from e
        return json.loads(raw)

    def _extract_text(self, data: dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Malformed response from server: {data}") from e

    def _coerce_json_text(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        return text.strip()

    def generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            "extra_body": {
                "top_k": self.top_k,
            },
        }
        data = self._post_json("/chat/completions", payload)
        return self._extract_text(data)

    def _request_json_once(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "extra_body": {
                "top_k": self.top_k,
            },
        }
        data = self._post_json("/chat/completions", payload)
        return self._coerce_json_text(self._extract_text(data))

    def generate_json(self, prompt: str) -> dict[str, Any]:
        strict_prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY one valid JSON object. "
              "No markdown, no explanation, no prose."
        )

        text = self._request_json_once(strict_prompt)
        try:
            return loads_maybe_json(text)
        except Exception:
            repair_prompt = (
                "You are given malformed JSON text. "
                "Rewrite it into ONE valid JSON object with the same information. "
                "Return ONLY JSON.\n\n"
                f"MALFORMED_JSON:\n{text}"
            )
            repaired = self._request_json_once(repair_prompt)
            try:
                return loads_maybe_json(repaired)
            except Exception:
                return json.loads(repaired)
