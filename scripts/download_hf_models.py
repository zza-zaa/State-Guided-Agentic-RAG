from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/model_manifest.yaml"))
    parser.add_argument("--profile", type=str, default="default")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/hf"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    profile = manifest["profiles"][args.profile]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    repos = []
    for _, model_list in profile.items():
        repos.extend(model_list)

    seen = set()
    for repo_id in repos:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        local_dir = args.cache_dir / repo_id.replace("/", "--")
        print(f"[download] {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


if __name__ == "__main__":
    main()
