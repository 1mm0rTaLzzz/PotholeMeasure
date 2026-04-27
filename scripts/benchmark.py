"""Benchmark multiple pothole instance-segmentation models in one run.

Adds paper-oriented protocol controls:
- repeated runs with aggregated mean/std/95% CI
- optional COCO metrics beyond headline mAP (APs/APm/APl)
- PR-curve points export
- runtime environment capture (CUDA, torch, driver)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--input-dir", type=Path, required=True, help="folder of test images")
    p.add_argument("--annotations", type=Path, default=None, help="COCO JSON for segm mAP (optional)")
    p.add_argument("--num", type=int, default=100, help="number of images to time (after warmup)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1, help="number of repeated benchmark runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("experiments/results/benchmark.json"))
    return p.parse_args()


def _summary(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "mean_ms": float(statistics.mean(samples) * 1000),
        "median_ms": float(statistics.median(samples) * 1000),
        "p95_ms": float(sorted(samples)[int(0.95 * len(samples)) - 1] * 1000) if len(samples) >= 20 else float(max(samples) * 1000),
        "min_ms": float(min(samples) * 1000),
        "max_ms": float(max(samples) * 1000),
    }


def _aggregate(repeat_values: list[float]) -> dict[str, float | int]:
    if not repeat_values:
        return {"n": 0}
    n = len(repeat_values)
    mean = statistics.mean(repeat_values)
    std = statistics.stdev(repeat_values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": float(mean), "std": float(std), "ci95": float(ci95)}


def _rle_encode(mask: "np.ndarray") -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(mask.astype("uint8", copy=False, order="F"))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _run_yolo(model_cfg: dict[str, Any], image: "np.ndarray") -> tuple[list[dict[str, Any]], float]:
    model = model_cfg["_model"]
    ts = time.perf_counter()
    results = model.predict(
        source=image,
        conf=float(model_cfg.get("conf", 0.25)),
        iou=float(model_cfg.get("iou", 0.5)),
        imgsz=int(model_cfg.get("imgsz", 1280)),
        device=str(model_cfg.get("device", "cuda")),
        verbose=False,
    )
    dt = time.perf_counter() - ts
    if not results:
        return [], dt
    r0 = results[0]
    if r0.masks is None or len(r0.masks) == 0:
        return [], dt
    masks = r0.masks.data.cpu().numpy().astype(bool)
    scores = r0.boxes.conf.cpu().numpy()
    labels = r0.boxes.cls.cpu().numpy().astype(int)
    return [
        {"mask": masks[i], "score": float(scores[i]), "category_id": int(labels[i]) + 1}
        for i in range(len(masks))
    ], dt


def _run_mmdet(model_cfg: dict[str, Any], image_path: Path) -> tuple[list[dict[str, Any]], float]:
    inferencer = model_cfg["_model"]
    ts = time.perf_counter()
    out = inferencer(str(image_path), return_vis=False, no_save_pred=True)
    dt = time.perf_counter() - ts

    preds = out.get("predictions", [])
    if not preds:
        return [], dt
    p0 = preds[0]
    masks = p0.get("masks", []) or []
    scores = p0.get("scores", []) or []
    labels = p0.get("labels", []) or []

    detections = []
    for m, s, c in zip(masks, scores, labels):
        if float(s) < float(model_cfg.get("conf", 0.25)):
            continue
        detections.append({"mask": m, "score": float(s), "category_id": int(c) + 1})
    return detections, dt


def _load_models(model_cfgs: list[dict[str, Any]]) -> None:
    from ultralytics import YOLO

    for cfg in model_cfgs:
        family = cfg["family"].lower()
        if family == "yolo":
            cfg["_model"] = YOLO(str(cfg["weights"]))
            continue
        if family == "mmdet":
            from mmdet.apis import DetInferencer

            cfg["_model"] = DetInferencer(
                model=str(cfg["config"]),
                weights=str(cfg["weights"]),
                device=str(cfg.get("device", "cuda")),
            )
            continue
        raise ValueError(f"unsupported model family: {family}")


def _apply_protocol(model_cfgs: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    allow_override = bool(protocol.get("allow_model_overrides", False))
    for cfg in model_cfgs:
        for key in ("conf", "iou", "imgsz", "device"):
            if key in protocol and (not allow_override or key not in cfg):
                cfg[key] = protocol[key]


def _environment_info() -> dict[str, Any]:
    import platform

    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return info


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2
    import numpy as np
    import yaml

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text())
    bench_cfg = cfg.get("benchmark", {})
    protocol = bench_cfg.get("protocol", {})
    model_cfgs: list[dict[str, Any]] = bench_cfg.get("models", [])
    if not model_cfgs:
        raise SystemExit("no benchmark.models found in config")
    _apply_protocol(model_cfgs, protocol)
    _load_models(model_cfgs)

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")
    images = images[: args.num + args.warmup]

    report: dict[str, Any] = {
        "config": str(args.config),
        "input_dir": str(args.input_dir),
        "protocol": protocol,
        "warmup": args.warmup,
        "num": args.num,
        "repeats": args.repeats,
        "seed": args.seed,
        "environment": _environment_info(),
        "models": [],
    }

    coco_gt = None
    if args.annotations:
        from pycocotools.coco import COCO

        coco_gt = COCO(str(args.annotations))

    for model_cfg in model_cfgs:
        per_repeat: list[dict[str, Any]] = []

        for rep in range(args.repeats):
            latencies: list[float] = []
            coco_results: list[dict[str, Any]] = []
            n_preds = 0

            for i, img_path in enumerate(images):
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                if model_cfg["family"].lower() == "yolo":
                    preds, dt = _run_yolo(model_cfg, image)
                else:
                    preds, dt = _run_mmdet(model_cfg, img_path)

                if i >= args.warmup:
                    latencies.append(dt)
                n_preds += len(preds)

                if coco_gt is not None:
                    img_id = None
                    for cid, meta in coco_gt.imgs.items():
                        if Path(meta["file_name"]).name == img_path.name:
                            img_id = cid
                            break
                    if img_id is None:
                        continue
                    for p in preds:
                        mask = np.asarray(p["mask"], dtype=np.uint8)
                        coco_results.append(
                            {
                                "image_id": img_id,
                                "category_id": p["category_id"],
                                "segmentation": _rle_encode(mask),
                                "score": float(p["score"]),
                            }
                        )

            rep_report: dict[str, Any] = {
                "repeat_index": rep,
                "latency": _summary(latencies),
                "fps_mean": (1.0 / statistics.mean(latencies)) if latencies else 0.0,
                "fps_median": (1.0 / statistics.median(latencies)) if latencies else 0.0,
                "num_predictions": n_preds,
            }

            if coco_gt is not None and coco_results:
                from pycocotools.cocoeval import COCOeval

                coco_dt = coco_gt.loadRes(coco_results)
                evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
                precision = evaluator.eval.get("precision")
                pr_curve = []
                if precision is not None and precision.ndim == 5:
                    # iou=0.5 (index 0), class=0, area=all(index 0), maxDets=100(index 2)
                    pr = precision[0, :, 0, 0, 2]
                    pr_curve = [float(x) if x >= 0 else None for x in pr.tolist()]

                rep_report["coco_segm"] = {
                    "mAP_50_95": float(evaluator.stats[0]),
                    "mAP_50": float(evaluator.stats[1]),
                    "mAP_75": float(evaluator.stats[2]),
                    "AP_small": float(evaluator.stats[3]),
                    "AP_medium": float(evaluator.stats[4]),
                    "AP_large": float(evaluator.stats[5]),
                    "pr_curve_iou50": pr_curve,
                }
            per_repeat.append(rep_report)

        model_report: dict[str, Any] = {
            "name": model_cfg["name"],
            "family": model_cfg["family"],
            "policy": {
                "conf": model_cfg.get("conf"),
                "iou": model_cfg.get("iou"),
                "imgsz": model_cfg.get("imgsz"),
                "device": model_cfg.get("device"),
            },
            "per_repeat": per_repeat,
            "aggregate": {
                "latency_mean_ms": _aggregate([float(r["latency"].get("mean_ms", 0.0)) for r in per_repeat]),
                "fps_mean": _aggregate([float(r.get("fps_mean", 0.0)) for r in per_repeat]),
                "mAP_50_95": _aggregate([float(r.get("coco_segm", {}).get("mAP_50_95", 0.0)) for r in per_repeat if "coco_segm" in r]),
                "mAP_50": _aggregate([float(r.get("coco_segm", {}).get("mAP_50", 0.0)) for r in per_repeat if "coco_segm" in r]),
            },
        }
        report["models"].append(model_report)

        lat = model_report["aggregate"]["latency_mean_ms"]
        fps = model_report["aggregate"]["fps_mean"]
        logging.info(
            "%s: latency_mean=%.1f±%.1f ms (95%%CI), fps=%.2f±%.2f",
            model_cfg["name"],
            lat.get("mean", 0.0),
            lat.get("ci95", 0.0),
            fps.get("mean", 0.0),
            fps.get("ci95", 0.0),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    logging.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
