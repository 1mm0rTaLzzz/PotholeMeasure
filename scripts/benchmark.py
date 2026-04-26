"""Measure inference speed of the full pothole pipeline.

Reports per-stage and end-to-end timings averaged over a folder of images.
The numbers are paper-friendly: mean / median / p95 latency in ms and the
corresponding FPS for the end-to-end loop.

Stages timed separately:
    seg     YOLOv8-seg forward pass
    depth   Depth Anything V2 metric forward pass
    geom    plane fit + per-mask depth offset + BEV area + classifier
    total   sum of the above plus any small overheads inside .process()

A short warm-up phase (default 3 images) is excluded from the report.

Example::

    python scripts/benchmark.py \\
        --config configs/default.yaml \\
        --input-dir data\\processed\\test \\
        --num 100 --output experiments\\results\\paper\\benchmark.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--input-dir", type=Path, required=True, help="folder of test images")
    p.add_argument("--num", type=int, default=100, help="number of images to time (after warmup)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("experiments/results/benchmark.json"))
    return p.parse_args()


def _summary(samples: list[float]) -> dict:
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2

    from src.geometry.homography import mask_to_area_m2
    from src.geometry.plane_fitting import compute_depth_offset, depth_to_pointcloud, fit_road_plane
    from src.pipeline import PotholePipeline

    pipeline = PotholePipeline.from_config(args.config)
    plane_cfg = pipeline.plane_cfg

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if not images:
        raise SystemExit(f"no images in {args.input_dir}")
    images = images[: args.num + args.warmup]

    seg_t: list[float] = []
    depth_t: list[float] = []
    geom_t: list[float] = []
    total_t: list[float] = []
    n_potholes_seen = 0

    for i, img_path in enumerate(images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # --- end-to-end ---
        t0 = time.perf_counter()
        # --- seg ---
        ts = time.perf_counter()
        detections = pipeline.segmentor.predict(image)
        seg_dt = time.perf_counter() - ts

        # --- depth ---
        ts = time.perf_counter()
        depth_map = pipeline.depth_estimator.predict(image)
        depth_dt = time.perf_counter() - ts

        # --- geom (plane + offset + area + classify) ---
        ts = time.perf_counter()
        h, w = depth_map.shape
        pc = depth_to_pointcloud(depth_map, pipeline.K)
        plane = fit_road_plane(
            pc, (h, w),
            exclude_masks=[d.mask for d in detections],
            threshold_m=float(plane_cfg.get("ransac_threshold_m", 0.02)),
            max_iters=int(plane_cfg.get("max_iterations", 1000)),
            min_inlier_ratio=float(plane_cfg.get("min_inlier_ratio", 0.7)),
            min_points=int(plane_cfg.get("min_points", 1000)),
            up_axis=tuple(plane_cfg.get("up_axis", (0.0, -1.0, 0.0))),
            up_cos_threshold=float(plane_cfg.get("up_cos_threshold", 0.9)),
        )
        for det in detections:
            if plane is not None:
                compute_depth_offset(plane, pc, det.mask)
            mask_to_area_m2(det.mask, pipeline.H_img2world)
            n_potholes_seen += 1
        geom_dt = time.perf_counter() - ts
        total_dt = time.perf_counter() - t0

        if i < args.warmup:
            continue
        seg_t.append(seg_dt)
        depth_t.append(depth_dt)
        geom_t.append(geom_dt)
        total_t.append(total_dt)

        if (len(total_t)) % 20 == 0:
            logging.info(
                "[%d/%d] total=%.1fms seg=%.1f depth=%.1f geom=%.1f",
                len(total_t), args.num, total_dt * 1000, seg_dt * 1000, depth_dt * 1000, geom_dt * 1000,
            )

    report = {
        "config": str(args.config),
        "input_dir": str(args.input_dir),
        "warmup": args.warmup,
        "n_potholes_total": n_potholes_seen,
        "stages": {
            "seg": _summary(seg_t),
            "depth": _summary(depth_t),
            "geom": _summary(geom_t),
            "total": _summary(total_t),
        },
        "fps_total_mean": (1.0 / statistics.mean(total_t)) if total_t else 0.0,
        "fps_total_median": (1.0 / statistics.median(total_t)) if total_t else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    logging.info("wrote %s", args.output)
    logging.info(
        "end-to-end: mean=%.1f ms (%.1f FPS), median=%.1f ms (%.1f FPS), p95=%.1f ms",
        report["stages"]["total"]["mean_ms"], report["fps_total_mean"],
        report["stages"]["total"]["median_ms"], report["fps_total_median"],
        report["stages"]["total"]["p95_ms"],
    )


if __name__ == "__main__":
    main()
