"""Create qualitative side-by-side comparisons for benchmark models."""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("experiments/results/paper/qualitative"))
    p.add_argument("--num-images", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_models(model_cfgs: list[dict[str, Any]]) -> None:
    from ultralytics import YOLO

    for cfg in model_cfgs:
        if cfg["family"].lower() == "yolo":
            cfg["_model"] = YOLO(str(cfg["weights"]))
        else:
            from mmdet.apis import DetInferencer

            cfg["_model"] = DetInferencer(model=str(cfg["config"]), weights=str(cfg["weights"]), device=str(cfg.get("device", "cuda")))


def _predict(cfg: dict[str, Any], image: np.ndarray, img_path: Path) -> list[dict[str, Any]]:
    if cfg["family"].lower() == "yolo":
        res = cfg["_model"].predict(source=image, conf=float(cfg.get("conf", 0.25)), iou=float(cfg.get("iou", 0.5)), imgsz=int(cfg.get("imgsz", 1280)), device=str(cfg.get("device", "cuda")), verbose=False)
        if not res or res[0].masks is None:
            return []
        masks = res[0].masks.data.cpu().numpy().astype(bool)
        scores = res[0].boxes.conf.cpu().numpy()
        return [{"mask": masks[i], "score": float(scores[i])} for i in range(len(masks))]

    out = cfg["_model"](str(img_path), return_vis=False, no_save_pred=True)
    preds = out.get("predictions", [])
    if not preds:
        return []
    p0 = preds[0]
    masks, scores = p0.get("masks", []) or [], p0.get("scores", []) or []
    return [{"mask": np.asarray(m, dtype=bool), "score": float(s)} for m, s in zip(masks, scores) if float(s) >= float(cfg.get("conf", 0.25))]


def _overlay(image: np.ndarray, preds: list[dict[str, Any]], color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    for p in preds:
        m = p["mask"].astype(bool)
        out[m] = (0.55 * out[m] + 0.45 * np.array(color)).astype(np.uint8)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text())
    protocol = cfg.get("benchmark", {}).get("protocol", {})
    model_cfgs: list[dict[str, Any]] = cfg.get("benchmark", {}).get("models", [])
    for m in model_cfgs:
        for k in ("conf", "iou", "imgsz", "device"):
            if k in protocol and k not in m:
                m[k] = protocol[k]

    _load_models(model_cfgs)

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if not images:
        raise SystemExit("No images found")
    chosen = random.sample(images, min(args.num_images, len(images)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    palette = [(255, 0, 0), (0, 180, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 128, 255)]

    cols = 3
    rows = math.ceil((len(model_cfgs) + 1) / cols)

    for img_path in chosen:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        panels = [image]
        labels = ["Input"]

        for i, model in enumerate(model_cfgs):
            preds = _predict(model, image, img_path)
            ov = _overlay(image, preds, palette[i % len(palette)])
            cv2.putText(ov, f"{model['name']} | n={len(preds)}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            panels.append(ov)
            labels.append(model["name"])

        h, w = image.shape[:2]
        cell = np.zeros((h, w, 3), dtype=np.uint8)
        while len(panels) < rows * cols:
            panels.append(cell.copy())

        grid_rows = []
        idx = 0
        for _ in range(rows):
            row = np.hstack(panels[idx: idx + cols])
            grid_rows.append(row)
            idx += cols
        canvas = np.vstack(grid_rows)

        out_path = args.output_dir / f"{img_path.stem}_grid.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
