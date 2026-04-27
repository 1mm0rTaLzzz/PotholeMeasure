"""Download/validate model assets for 6-model benchmark and write lock file.

Config format (`benchmark.models` item):
  - local path in `weights` (and optional `config` for mmdet), OR
  - `download` section:
      source: huggingface | url
      repo_id: ...            # for huggingface
      filename: ...           # for huggingface
      url: ...                # for url
      out: path/to/file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--lock-file", type=Path, default=Path("experiments/checkpoints/benchmark_lock.json"))
    p.add_argument("--strict", action="store_true", help="fail if any required file is missing")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_hf(repo_id: str, filename: str, out: Path) -> Path:
    from huggingface_hub import hf_hub_download

    out.parent.mkdir(parents=True, exist_ok=True)
    local = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(out.parent), local_dir_use_symlinks=False))
    if local != out:
        out.write_bytes(local.read_bytes())
    return out


def _download_url(url: str, out: Path) -> Path:
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(out))
    return out


def _pip_freeze_subset() -> list[str]:
    pkgs = ["torch", "torchvision", "ultralytics", "mmdet", "mmengine", "mmcv", "pycocotools", "opencv-python", "numpy"]
    result = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    out = []
    for ln in lines:
        low = ln.lower()
        if any(low.startswith(f"{p.lower()}==") for p in pkgs):
            out.append(ln)
    return sorted(out)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    models = cfg.get("benchmark", {}).get("models", [])
    if not models:
        raise SystemExit("No benchmark.models entries found")

    records: list[dict[str, Any]] = []
    missing: list[str] = []

    for model in models:
        item: dict[str, Any] = {"name": model.get("name"), "family": model.get("family")}
        weights = Path(model["weights"])

        dl = model.get("download")
        if dl:
            src = dl.get("source")
            out = Path(dl.get("out", weights))
            if src == "huggingface":
                weights = _download_hf(dl["repo_id"], dl["filename"], out)
            elif src == "url":
                weights = _download_url(dl["url"], out)
            model["weights"] = str(weights)

        item["weights"] = str(weights)
        if weights.exists():
            item["weights_sha256"] = sha256(weights)
            item["weights_size"] = weights.stat().st_size
        else:
            missing.append(str(weights))

        if model.get("family", "").lower() == "mmdet":
            cfg_path = Path(model["config"])
            item["config"] = str(cfg_path)
            if cfg_path.exists():
                item["config_sha256"] = sha256(cfg_path)
            else:
                missing.append(str(cfg_path))

        records.append(item)

    lock = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": _pip_freeze_subset(),
        "models": records,
        "missing": missing,
    }
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.lock_file.write_text(json.dumps(lock, indent=2))
    print(f"Wrote lock file: {args.lock_file}")

    if args.strict and missing:
        raise SystemExit("Missing required model assets:\n- " + "\n- ".join(missing))


if __name__ == "__main__":
    main()
