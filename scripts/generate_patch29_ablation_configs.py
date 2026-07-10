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
    out = Path(f"configs/ablations/{name}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[done] wrote {out}")

# A0 full
cfg = copy.deepcopy(base)
dump("patch29_full_14b", cfg)

# A1 w/o typed question specification
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_question_spec"] = False
dump("patch29_no_typed_spec_14b", cfg)

# A2 w/o typed-guided second-hop
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_second_hop"] = False
dump("patch29_no_second_hop_14b", cfg)

# A3 w/o typed uncertainty retry
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_typed_retry"] = False
dump("patch29_no_typed_retry_14b", cfg)

# A4 w/o dependency-conditioned query
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_dependency_query"] = False
dump("patch29_no_dependency_query_14b", cfg)

# A5 w/o title hop
cfg = copy.deepcopy(base)
p = ensure_pipeline(cfg)
p["enable_title_hop"] = False
dump("patch29_no_title_hop_14b", cfg)
