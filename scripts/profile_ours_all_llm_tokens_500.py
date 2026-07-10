from __future__ import annotations

import argparse
import json
import time
import statistics
import traceback
from pathlib import Path
from typing import Any
from collections import defaultdict

from transformers import AutoTokenizer


class TokenMeter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.current_key = None
        self.events = defaultdict(list)

    def start(self, key: str):
        self.current_key = key

    def end(self):
        self.current_key = None

    def count_text(self, text: str) -> int:
        if text is None:
            return 0
        text = str(text)
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return 0

    def count_token_ids(self, token_ids) -> int:
        if token_ids is None:
            return 0
        try:
            return len(token_ids)
        except Exception:
            return 0

    def record(self, prompt_tokens: int, generation_tokens: int, source: str, prompt_preview: str = "", output_preview: str = ""):
        if self.current_key is None:
            return

        prompt_tokens = int(prompt_tokens or 0)
        generation_tokens = int(generation_tokens or 0)

        self.events[self.current_key].append({
            "source": source,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "total_tokens": prompt_tokens + generation_tokens,
            "prompt_preview": str(prompt_preview)[:300] if prompt_preview else "",
            "output_preview": str(output_preview)[:300] if output_preview else "",
        })

    def summary_for(self, key: str):
        evs = self.events.get(key, [])
        prompt = sum(e["prompt_tokens"] for e in evs)
        generation = sum(e["generation_tokens"] for e in evs)
        total = sum(e["total_tokens"] for e in evs)
        return {
            "llm_call_count": len(evs),
            "prompt_tokens": prompt,
            "generation_tokens": generation,
            "total_tokens": total,
            "llm_events": evs,
        }


METER = None


def normalize_prompts_from_vllm_args(args, kwargs):
    prompts = None

    if "prompts" in kwargs:
        prompts = kwargs.get("prompts")
    elif "prompt" in kwargs:
        prompts = kwargs.get("prompt")
    elif len(args) >= 1:
        prompts = args[0]

    prompt_token_ids = kwargs.get("prompt_token_ids", None)

    return prompts, prompt_token_ids


def count_vllm_prompts(prompts, prompt_token_ids):
    global METER

    prompt_tokens = 0
    prompt_preview = ""

    if isinstance(prompts, str):
        prompt_tokens += METER.count_text(prompts)
        prompt_preview = prompts[:300]
    elif isinstance(prompts, list):
        for p in prompts:
            if isinstance(p, str):
                prompt_tokens += METER.count_text(p)
                if not prompt_preview:
                    prompt_preview = p[:300]
            elif isinstance(p, dict):
                # Some vLLM inputs can be dict-like prompt objects.
                txt = p.get("prompt") or p.get("text") or ""
                if txt:
                    prompt_tokens += METER.count_text(txt)
                    if not prompt_preview:
                        prompt_preview = txt[:300]

    # Fallback: if text prompt is unavailable, use prompt_token_ids length.
    if prompt_tokens == 0 and prompt_token_ids is not None:
        if isinstance(prompt_token_ids, list):
            if prompt_token_ids and isinstance(prompt_token_ids[0], list):
                prompt_tokens = sum(len(x) for x in prompt_token_ids)
            else:
                prompt_tokens = len(prompt_token_ids)

    return prompt_tokens, prompt_preview


def patch_vllm_generate():
    try:
        import vllm
    except Exception as e:
        print(f"[warn] vLLM import failed, skip vLLM patch: {e}")
        return False

    try:
        orig_generate = vllm.LLM.generate
    except Exception as e:
        print(f"[warn] vLLM LLM.generate not found: {e}")
        return False

    if getattr(orig_generate, "_csa_all_token_patched", False):
        return True

    def patched_generate(self, *args, **kwargs):
        global METER

        prompts, prompt_token_ids = normalize_prompts_from_vllm_args(args, kwargs)
        prompt_tokens_from_args, prompt_preview = count_vllm_prompts(prompts, prompt_token_ids)

        outputs = orig_generate(self, *args, **kwargs)

        total_prompt_tokens = 0
        total_generation_tokens = 0
        output_preview = ""

        out_list = outputs if isinstance(outputs, (list, tuple)) else [outputs]

        for req_out in out_list:
            # vLLM usually exposes prompt_token_ids; use it if available because it is exact.
            ptids = getattr(req_out, "prompt_token_ids", None)
            if ptids is not None:
                total_prompt_tokens += len(ptids)

            gens = getattr(req_out, "outputs", None)
            if gens:
                for gen in gens:
                    text = getattr(gen, "text", "")
                    if text and not output_preview:
                        output_preview = text[:300]

                    # Prefer tokenizer-estimated text count for your requested definition.
                    # Fallback to generated token_ids length.
                    if text:
                        total_generation_tokens += METER.count_text(text)
                    else:
                        tids = getattr(gen, "token_ids", None)
                        total_generation_tokens += METER.count_token_ids(tids)

        # If request output does not expose prompt ids, use prompt string estimate.
        if total_prompt_tokens == 0:
            total_prompt_tokens = prompt_tokens_from_args

        METER.record(
            total_prompt_tokens,
            total_generation_tokens,
            "vllm.LLM.generate",
            prompt_preview=prompt_preview,
            output_preview=output_preview,
        )

        return outputs

    patched_generate._csa_all_token_patched = True
    vllm.LLM.generate = patched_generate
    print("[patch] vLLM LLM.generate patched for all-call tokenizer-estimated token counting")
    return True


def tensor_token_count(x):
    try:
        shape = tuple(x.shape)
        if len(shape) == 1:
            return int(shape[-1])
        if len(shape) >= 2:
            return int(shape[0] * shape[-1])
    except Exception:
        pass
    return 0


def patch_transformers_generate():
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:
        print(f"[warn] transformers GenerationMixin import failed, skip HF patch: {e}")
        return False

    orig_generate = GenerationMixin.generate
    if getattr(orig_generate, "_csa_all_token_patched", False):
        return True

    def patched_generate(self, *args, **kwargs):
        global METER

        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            first = args[0]
            if hasattr(first, "shape"):
                input_ids = first

        prompt_tokens = tensor_token_count(input_ids)
        outputs = orig_generate(self, *args, **kwargs)

        total_output_tokens = 0
        if hasattr(outputs, "sequences"):
            total_output_tokens = tensor_token_count(outputs.sequences)
        elif hasattr(outputs, "shape"):
            total_output_tokens = tensor_token_count(outputs)

        generation_tokens = max(total_output_tokens - prompt_tokens, 0)

        METER.record(
            prompt_tokens,
            generation_tokens,
            "transformers.generate",
        )

        return outputs

    patched_generate._csa_all_token_patched = True
    GenerationMixin.generate = patched_generate
    print("[patch] transformers.generate patched for token counting")
    return True


def serialize_openai_messages(messages):
    if not messages:
        return ""
    try:
        return json.dumps(messages, ensure_ascii=False)
    except Exception:
        return str(messages)


def patch_openai_chat_create():
    try:
        from openai.resources.chat.completions.completions import Completions
    except Exception as e:
        print(f"[warn] OpenAI Completions import failed, skip OpenAI patch: {e}")
        return False

    orig_create = Completions.create
    if getattr(orig_create, "_csa_all_token_patched", False):
        return True

    def patched_create(self, *args, **kwargs):
        global METER

        messages = kwargs.get("messages", None)
        prompt_text = serialize_openai_messages(messages)
        prompt_tokens_est = METER.count_text(prompt_text)

        res = orig_create(self, *args, **kwargs)

        output_text = ""
        try:
            choices = getattr(res, "choices", []) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", "")
                    output_text = content or ""
        except Exception:
            output_text = ""

        generation_tokens_est = METER.count_text(output_text)

        usage = getattr(res, "usage", None)
        logged_prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        logged_completion = getattr(usage, "completion_tokens", None) if usage is not None else None

        # Prefer real usage if available; otherwise tokenizer-estimated message/output text.
        prompt_tokens = logged_prompt if logged_prompt is not None else prompt_tokens_est
        generation_tokens = logged_completion if logged_completion is not None else generation_tokens_est

        METER.record(
            prompt_tokens,
            generation_tokens,
            "openai.chat.completions.create",
            prompt_preview=prompt_text,
            output_preview=output_text,
        )

        return res

    patched_create._csa_all_token_patched = True
    Completions.create = patched_create
    print("[patch] OpenAI chat.completions.create patched for token counting")
    return True


def load_rows(path: Path):
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset file: {path}")


def get_question(row: dict) -> str:
    q = row.get("question") or row.get("query")
    if not q:
        raise ValueError(f"Missing question field: {row.keys()}")
    return q


def get_qid(row: dict, idx: int) -> str:
    return str(row.get("_id") or row.get("id") or row.get("question_id") or row.get("qid") or f"row_{idx}")


def to_plain(obj: Any):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def infer_steps(obj: Any):
    if isinstance(obj, dict):
        for k in ["steps", "num_steps", "n_steps", "step_count", "avg_steps"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list):
                    return float(len(v))
        for k in ["trajectory", "history", "actions", "state_history", "reasoning_steps"]:
            if k in obj and isinstance(obj[k], list):
                return float(len(obj[k]))
        for v in obj.values():
            r = infer_steps(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for x in obj:
            r = infer_steps(x)
            if r is not None:
                return r
    return None


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p))
    return xs[idx]


def avg(xs):
    vals = [x for x in xs if isinstance(x, (int, float))]
    return sum(vals) / len(vals) if vals else None


def main():
    global METER

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--tokenizer-path", default="/mnt/raid/zsb/llm_models/Qwen3-14B")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--output-summary", required=True)
    ap.add_argument("--output-details", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    METER = TokenMeter(tokenizer)

    # Patch before importing / instantiating pipeline.
    patch_vllm_generate()
    patch_transformers_generate()
    patch_openai_chat_create()

    from csa_rag.agent.pipeline import CSARAGPipeline

    dataset_path = Path(args.dataset_path)
    rows = load_rows(dataset_path)
    rows = rows[args.offset: args.offset + args.limit]

    t0 = time.perf_counter()
    pipe = CSARAGPipeline(
        models_config=Path(args.models_config),
        index_dir=Path(args.index_dir),
    )
    t1 = time.perf_counter()
    init_sec = t1 - t0

    details = []

    elapsed_ok = []
    elapsed_all = []
    steps_ok = []

    prompt_tok_ok = []
    generation_tok_ok = []
    total_tok_ok = []
    llm_calls_ok = []

    ok = 0
    fail = 0

    for local_i, row in enumerate(rows, 1):
        global_i = args.offset + local_i
        qid = get_qid(row, global_i)
        question = get_question(row)
        key = f"{args.dataset_name}:{global_i}:{qid}"

        METER.start(key)
        qs = time.perf_counter()

        try:
            result = pipe.run(question)
            qe = time.perf_counter()
            elapsed = qe - qs

            METER.end()

            plain = to_plain(result)
            step_val = infer_steps(plain)
            token_info = METER.summary_for(key)

            ok += 1
            elapsed_ok.append(elapsed)
            elapsed_all.append(elapsed)

            if step_val is not None:
                steps_ok.append(step_val)

            prompt_tok_ok.append(token_info["prompt_tokens"])
            generation_tok_ok.append(token_info["generation_tokens"])
            total_tok_ok.append(token_info["total_tokens"])
            llm_calls_ok.append(token_info["llm_call_count"])

            details.append({
                "idx": local_i,
                "global_idx": global_i,
                "qid": qid,
                "status": "ok",
                "elapsed_sec_excl_init": elapsed,
                "steps": step_val,
                "llm_call_count": token_info["llm_call_count"],
                "prompt_tokens_all_llm_calls": token_info["prompt_tokens"],
                "generation_tokens_all_llm_calls": token_info["generation_tokens"],
                "total_tokens_all_llm_calls": token_info["total_tokens"],
                "llm_events": token_info["llm_events"],
            })

        except Exception as e:
            qe = time.perf_counter()
            elapsed = qe - qs
            METER.end()

            fail += 1
            elapsed_all.append(elapsed)

            token_info = METER.summary_for(key)

            details.append({
                "idx": local_i,
                "global_idx": global_i,
                "qid": qid,
                "status": "error",
                "elapsed_sec_excl_init": elapsed,
                "llm_call_count": token_info["llm_call_count"],
                "prompt_tokens_all_llm_calls": token_info["prompt_tokens"],
                "generation_tokens_all_llm_calls": token_info["generation_tokens"],
                "total_tokens_all_llm_calls": token_info["total_tokens"],
                "llm_events": token_info["llm_events"],
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    summary = {
        "dataset": args.dataset_name,
        "method": "Ours",
        "model": "Qwen3-14B",
        "dataset_path": str(dataset_path),
        "index_dir": args.index_dir,
        "models_config": args.models_config,
        "limit": args.limit,
        "offset": args.offset,
        "init_sec_excluded": init_sec,
        "success_count": ok,
        "fail_count": fail,
        "avg_time_sec_per_question_success_only_excl_init": avg(elapsed_ok),
        "p50_time_sec_excl_init": statistics.median(elapsed_ok) if elapsed_ok else None,
        "p90_time_sec_excl_init": percentile(elapsed_ok, 0.90),
        "avg_time_sec_all_attempts_excl_init": avg(elapsed_all),
        "avg_steps_success_only": avg(steps_ok),
        "avg_llm_call_count_success_only": avg(llm_calls_ok),
        "avg_prompt_tokens_all_llm_calls": avg(prompt_tok_ok),
        "avg_generation_tokens_all_llm_calls": avg(generation_tok_ok),
        "avg_total_tokens_all_llm_calls": avg(total_tok_ok),
        "tokenizer_path": args.tokenizer_path,
        "token_note": (
            "Token counts are tokenizer-estimated with the Qwen3 tokenizer over all LLM calls. "
            "For each question, every prompt input and model-generated output across the full "
            "inference process is tokenized and summed; the reported values are averages over "
            "successful questions."
        ),
    }

    out_summary = Path(args.output_summary)
    out_details = Path(args.output_details)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_details.parent.mkdir(parents=True, exist_ok=True)

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out_details.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if ok > 0 and avg(llm_calls_ok) == 0:
        print("[warning] No LLM calls were intercepted. The pipeline may use an unsupported LLM wrapper; inspect details/logs.")


if __name__ == "__main__":
    main()
