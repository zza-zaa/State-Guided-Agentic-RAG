from __future__ import annotations

import json
from typing import Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from csa_rag.llm.base import BaseLLM
from csa_rag.utils.json_utils import loads_maybe_json


class VLLMLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 16384,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 1024,
        thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(model=model_name, max_model_len=max_model_len, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        self.repair_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_new_tokens,
        )
        self.thinking = thinking

    def _render_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def generate_text(self, prompt: str, repair: bool = False) -> str:
        rendered = self._render_chat(prompt)
        params = self.repair_sampling_params if repair else self.sampling_params
        outputs = self.llm.generate([rendered], params)
        return outputs[0].outputs[0].text

    def generate_json(self, prompt: str) -> dict[str, Any]:
        last_exc: Exception | None = None
        last_text = ""

        candidate_prompts = [
            prompt,
            prompt + "\n\nIMPORTANT: Return STRICT JSON only. No markdown. No explanation.",
        ]

        for p in candidate_prompts:
            text = self.generate_text(p)
            last_text = text
            try:
                return loads_maybe_json(text)
            except Exception as e:
                last_exc = e

        repair_prompt = (
            "Rewrite the following content into ONE strict valid JSON object only.\n"
            "Do not add explanation.\n\n"
            f"{last_text}"
        )
        repair_text = self.generate_text(repair_prompt, repair=True)
        try:
            return loads_maybe_json(repair_text)
        except Exception as e:
            last_exc = e

        raise last_exc if last_exc is not None else ValueError("generate_json failed")
