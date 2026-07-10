from __future__ import annotations

import argparse
import json
import time
import statistics
import traceback
from pathlib import Path
from typing import Any

import yaml

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

from csa_rag.agent.pipeline import CSARAGPipeline


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
    raise ValueError(f"Unsupported JSON format: {path}")


def get_question(row: dict) -> str:
    q = row.get("question")
    if not q:
        raise ValueError(f"Missing question field: {row.keys()}")
    return q


def get_qid(row: dict, idx: int) -> str:
    return str(row.get("_id") or row.get("id") or row.get("question_id") or f"row_{idx}")


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


def recursive_find_numbers(obj: Any, keys):
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in keys and isinstance(v, (int, float)):
                vals.append(float(v))
            vals.extend(recursive_find_numbers(v, keys))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(recursive_find_numbers(x, keys))
    return vals


def recursive_collect_strings(obj: Any, key_hints):
    vals = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(h in lk for h in key_hints):
                if isinstance(v, str):
                    vals.append(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str):
                            vals.append(x)
                        elif isinstance(x, dict):
                            vals.extend(recursive_collect_strings(x, key_hints))
            vals.extend(recursive_collect_strings(v, key_hints))
    elif isinstance(obj, list):
        for x in obj:
            vals.extend(recursive_collect_strings(x, key_hints))
    return vals


def find_answer(obj: Any) -> str:
    if isinstance(obj, dict):
        for k in ["final_answer", "answer", "prediction", "pred"]:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        for v in obj.values():
            ans = find_answer(v)
            if ans:
                return ans
    elif isinstance(obj, list):
        for x in obj:
            ans = find_answer(x)
            if ans:
                return ans
    return ""


def infer_steps(obj: Any):
    if isinstance(obj, dict):
        for k in ["steps", "num_steps", "n_steps", "step_count", "avg_steps"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list):
                    return float(len(v))
        # 常见字段：trajectory/history/actions
        for k in ["trajectory", "history", "actions", "state_history", "reasoning_steps"]:
            if k in obj and isinstance(obj[k], list):
                return float(len(obj[k]))
        for v in obj.values():
            r = infer_steps(v)
            if r is not None:
                return r
    return None


def guess_tokenizer_path(models_config: Path):
    cfg = yaml.safe_load(models_config.read_text(encoding="utf-8"))
    candidates = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if "Qwen" in x or "qwen" in x:
                candidates.append(x)

    walk(cfg)
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return None


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None or not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * p))
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--output-summary", required=True)
    ap.add_argument("--output-details", required=True)
    ap.add_argument("--tokenizer-path", default=None)
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    index_dir = Path(args.index_dir)
    models_config = Path(args.models_config)

    rows = load_rows(dataset_path)
    rows = rows[args.offset: args.offset + args.limit]

    tok_path = args.tokenizer_path or guess_tokenizer_path(models_config)
    tokenizer = None
    if AutoTokenizer is not None and tok_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            print(f"[warn] failed to load tokenizer from {tok_path}: {e}")

    t0 = time.perf_counter()
    pipe = CSARAGPipeline(models_config=models_config, index_dir=index_dir)
    t1 = time.perf_counter()
    init_sec = t1 - t0

    details = []
    elapsed_ok = []
    elapsed_all = []
    step_vals = []
    input_tok_vals = []
    output_tok_vals = []
    total_tok_vals = []
    prompt_tok_logged = []
    completion_tok_logged = []
    total_tok_logged = []

    ok = 0
    fail = 0

    for i, row in enumerate(rows, 1):
        qid = get_qid(row, args.offset + i)
        q = get_question(row)

        qs = time.perf_counter()
        try:
            result = pipe.run(q)
            qe = time.perf_counter()
            elapsed = qe - qs
            elapsed_ok.append(elapsed)
            elapsed_all.append(elapsed)
            ok += 1

            plain = to_plain(result)
            steps = infer_steps(plain)
            if steps is not None:
                step_vals.append(steps)

            # 如果 pipeline 里记录了真实 token usage，就优先收集
            p_logged = recursive_find_numbers(
                plain,
                {"prompt_tokens", "input_tokens", "num_prompt_tokens"}
            )
            c_logged = recursive_find_numbers(
                plain,
                {"completion_tokens", "output_tokens", "num_completion_tokens"}
            )
            t_logged = recursive_find_numbers(
                plain,
                {"total_tokens", "num_total_tokens"}
            )

            if p_logged:
                prompt_tok_logged.append(sum(p_logged))
            if c_logged:
                completion_tok_logged.append(sum(c_logged))
            if t_logged:
                total_tok_logged.append(sum(t_logged))

            # 估计 token：question + exposed evidence/context + final answer
            ans = find_answer(plain)
            evidence_strings = recursive_collect_strings(
                plain,
                ["evidence", "context", "passage", "chunk", "document", "retrieved", "fact"]
            )
            # 防止递归重复太多，只取前 50 段字符串估计
            evidence_text = "\n".join(evidence_strings[:50])

            q_tokens = count_tokens(tokenizer, q)
            evidence_tokens = count_tokens(tokenizer, evidence_text)
            answer_tokens = count_tokens(tokenizer, ans)

            est_input = q_tokens + evidence_tokens
            est_output = answer_tokens
            est_total = est_input + est_output

            input_tok_vals.append(est_input)
            output_tok_vals.append(est_output)
            total_tok_vals.append(est_total)

            details.append({
                "idx": i,
                "qid": qid,
                "status": "ok",
                "elapsed_sec_excl_init": elapsed,
                "steps": steps,
                "question_tokens": q_tokens,
                "estimated_input_tokens": est_input,
                "estimated_output_tokens": est_output,
                "estimated_total_tokens": est_total,
                "logged_prompt_tokens_sum": sum(p_logged) if p_logged else None,
                "logged_completion_tokens_sum": sum(c_logged) if c_logged else None,
                "logged_total_tokens_sum": sum(t_logged) if t_logged else None,
                "answer_preview": ans[:200],
            })

        except Exception as e:
            qe = time.perf_counter()
            elapsed = qe - qs
            elapsed_all.append(elapsed)
            fail += 1
            details.append({
                "idx": i,
                "qid": qid,
                "status": "error",
                "elapsed_sec_excl_init": elapsed,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {
        "dataset": args.dataset_name,
        "dataset_path": str(dataset_path),
        "index_dir": str(index_dir),
        "models_config": str(models_config),
        "limit": args.limit,
        "offset": args.offset,
        "init_sec_excluded": init_sec,
        "success_count": ok,
        "fail_count": fail,
        "avg_time_sec_per_question_success_only_excl_init": avg(elapsed_ok),
        "p50_time_sec_excl_init": statistics.median(elapsed_ok) if elapsed_ok else None,
        "p90_time_sec_excl_init": percentile(elapsed_ok, 0.90),
        "avg_time_sec_all_attempts_excl_init": avg(elapsed_all),
        "avg_steps_success_only": avg(step_vals),
        "avg_estimated_input_tokens": avg(input_tok_vals),
        "avg_estimated_output_tokens": avg(output_tok_vals),
        "avg_estimated_total_tokens": avg(total_tok_vals),
        "avg_logged_prompt_tokens": avg(prompt_tok_logged),
        "avg_logged_completion_tokens": avg(completion_tok_logged),
        "avg_logged_total_tokens": avg(total_tok_logged),
        "tokenizer_path": tok_path,
        "token_note": "logged_* fields are real if the pipeline exposes token usage; estimated_* fields are tokenizer estimates from exposed question/evidence/answer strings.",
    }

    out_summary = Path(args.output_summary)
    out_details = Path(args.output_details)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_details.parent.mkdir(parents=True, exist_ok=True)

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_details.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
