from __future__ import annotations

from csa_rag.llm.vllm_llm import VLLMLLM
from csa_rag.llm.transformers_llm import TransformersLLM
from csa_rag.llm.openai_compat_llm import OpenAICompatLLM


def build_llm(cfg: dict):
    llm_cfg = cfg["llm"]
    runtime_cfg = cfg["runtime"]
    backend = llm_cfg.get("backend") or runtime_cfg.get("llm_backend", "vllm")

    if backend == "openai_compat":
        return OpenAICompatLLM(
            base_url=llm_cfg["server_base_url"],
            api_key=llm_cfg.get("server_api_key", "EMPTY"),
            model_name=llm_cfg.get("server_model_name", llm_cfg["default_model"]),
            temperature=llm_cfg.get("temperature", 0.6),
            top_p=llm_cfg.get("top_p", 0.95),
            top_k=llm_cfg.get("top_k", 20),
            max_new_tokens=llm_cfg.get("max_new_tokens", 1024),
            thinking=llm_cfg.get("thinking", True),
            timeout=llm_cfg.get("server_timeout", 600),
        )

    if runtime_cfg.get("use_vllm", True):
        return VLLMLLM(
            model_name=llm_cfg["default_model"],
            max_model_len=llm_cfg.get("max_model_len", 16384),
            temperature=llm_cfg.get("temperature", 0.6),
            top_p=llm_cfg.get("top_p", 0.95),
            top_k=llm_cfg.get("top_k", 20),
            max_new_tokens=llm_cfg.get("max_new_tokens", 1024),
            thinking=llm_cfg.get("thinking", True),
        )
    return TransformersLLM(
        model_name=llm_cfg["default_model"],
        max_new_tokens=llm_cfg.get("max_new_tokens", 1024),
        thinking=llm_cfg.get("thinking", True),
    )
