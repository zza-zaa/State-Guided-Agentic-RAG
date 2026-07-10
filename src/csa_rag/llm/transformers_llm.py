from __future__ import annotations

import json
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csa_rag.llm.base import BaseLLM
from csa_rag.utils.json_utils import loads_maybe_json


class TransformersLLM(BaseLLM):
    def __init__(self, model_name: str, max_new_tokens: int = 1024, thinking: bool = True) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.thinking = thinking

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        return self.tokenizer(rendered, return_tensors="pt").to(self.model.device)

    def generate_text(self, prompt: str) -> str:
        inputs = self._encode(prompt)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        completion = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        text = self.generate_text(prompt)
        try:
            return loads_maybe_json(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise
