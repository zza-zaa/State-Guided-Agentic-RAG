from pathlib import Path
import copy
import yaml

base_path = Path("configs/main_ablations/main_full_14b.yaml")
if not base_path.exists():
    base_path = Path("configs/models.yaml")

base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

steps = [1, 2, 3, 4, 5, 7, 9]
out_dir = Path("configs/fixed_steps")
out_dir.mkdir(parents=True, exist_ok=True)

def ensure_pipeline(cfg):
    if "pipeline" not in cfg or cfg["pipeline"] is None:
        cfg["pipeline"] = {}
    return cfg["pipeline"]

def ensure_agent(cfg):
    if "agent" not in cfg or cfg["agent"] is None:
        cfg["agent"] = {}
    return cfg["agent"]

for s in steps:
    cfg = copy.deepcopy(base)
    p = ensure_pipeline(cfg)
    a = ensure_agent(cfg)

    # Full model switches
    p["enable_typed_question_spec"] = True
    p["enable_second_hop"] = True
    p["enable_dependency_query"] = True
    p["enable_title_hop"] = True
    p["enable_typed_retry"] = True
    p["enable_typed_answer_grounding"] = True
    p["enable_typed_target_selector"] = False

    # Fixed-step controls
    p["fixed_steps"] = s
    p["fixed_step_limit"] = s
    p["fixed_max_steps"] = s
    p["max_steps"] = s
    p["max_reasoning_steps"] = s
    p["max_iterations"] = s
    p["step_budget"] = s
    p["num_steps"] = s

    # Force fixed budget: disable confidence early-stop.
    p["force_fixed_steps"] = True
    p["disable_early_stop"] = True
    p["enable_early_stop"] = False

    # Also synchronize agent.max_steps because older code paths may read this.
    a["max_steps"] = s

    out = out_dir / f"main_full_14b_fixedstep{s}.yaml"
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[done] wrote {out}")
