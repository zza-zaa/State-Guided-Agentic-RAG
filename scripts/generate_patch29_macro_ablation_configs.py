from pathlib import Path
import yaml
import copy

base_path = Path("configs/models.yaml")
base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

def ensure_pipeline(cfg):
    if "pipeline" not in cfg or cfg["pipeline"] is None:
        cfg["pipeline"] = {}
    return cfg["pipeline"]

def set_flags(cfg, typed, dep, title, second, retry):
    p = ensure_pipeline(cfg)
    p["enable_typed_question_spec"] = typed
    p["enable_dependency_query"] = dep
    p["enable_title_hop"] = title
    p["enable_second_hop"] = second
    p["enable_typed_retry"] = retry

def dump(name, cfg):
    out = Path(f"configs/macro_ablations/{name}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[done] wrote {out}")

# S1: CSA-Base
cfg = copy.deepcopy(base)
set_flags(cfg, typed=False, dep=False, title=False, second=False, retry=False)
dump("patch29_stage1_base_14b", cfg)

# S2: + Typed Understanding
cfg = copy.deepcopy(base)
set_flags(cfg, typed=True, dep=False, title=False, second=False, retry=False)
dump("patch29_stage2_typed_14b", cfg)

# S3: + Controlled Multi-hop Retrieval
cfg = copy.deepcopy(base)
set_flags(cfg, typed=True, dep=True, title=True, second=True, retry=False)
dump("patch29_stage3_multihop_14b", cfg)

# S4: Full
cfg = copy.deepcopy(base)
set_flags(cfg, typed=True, dep=True, title=True, second=True, retry=True)
dump("patch29_stage4_full_14b", cfg)
