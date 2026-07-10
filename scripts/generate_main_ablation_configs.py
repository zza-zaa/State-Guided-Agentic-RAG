from pathlib import Path
import yaml
import copy

base_path = Path("configs/models.yaml")
base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

def ensure_pipeline(cfg):
    if "pipeline" not in cfg or cfg["pipeline"] is None:
        cfg["pipeline"] = {}
    return cfg["pipeline"]

def dump(name, cfg):
    out = Path(f"configs/main_ablations/{name}.yaml")
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[done] wrote {out}")

# ------------------------------------------------------------
# M0: State Only
# Keep explicit state / slot pipeline, but disable:
# - question type optimization
# - state-based retrieval optimization
# - retry / answer grounding / target selector
# ------------------------------------------------------------
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_question_spec"] = False
p["enable_second_hop"] = False
p["enable_dependency_query"] = False
p["enable_title_hop"] = False
p["enable_typed_retry"] = False
p["enable_typed_answer_grounding"] = False
p["enable_typed_target_selector"] = False
dump("main_state_only_14b", cfg)

# ------------------------------------------------------------
# M1: w/o Question Type Optimization
# Keep state and state-based optimization, remove typed question spec.
# ------------------------------------------------------------
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_question_spec"] = False
p["enable_second_hop"] = True
p["enable_dependency_query"] = True
p["enable_title_hop"] = True
p["enable_typed_retry"] = True
p["enable_typed_answer_grounding"] = True
p["enable_typed_target_selector"] = False
dump("main_no_question_type_14b", cfg)

# ------------------------------------------------------------
# M2: w/o State-based Optimization
# Keep question type optimization and state backbone,
# but remove optimizations built on state:
# - typed second-hop
# - dependency-conditioned query
# - title hop
# - typed retry
# - answer grounding / target selector
# ------------------------------------------------------------
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_question_spec"] = True
p["enable_second_hop"] = False
p["enable_dependency_query"] = False
p["enable_title_hop"] = False
p["enable_typed_retry"] = False
p["enable_typed_answer_grounding"] = False
p["enable_typed_target_selector"] = False
dump("main_no_state_optimization_14b", cfg)

# ------------------------------------------------------------
# M3: Full Model
# ------------------------------------------------------------
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_question_spec"] = True
p["enable_second_hop"] = True
p["enable_dependency_query"] = True
p["enable_title_hop"] = True
p["enable_typed_retry"] = True
p["enable_typed_answer_grounding"] = True
p["enable_typed_target_selector"] = False
dump("main_full_14b", cfg)
