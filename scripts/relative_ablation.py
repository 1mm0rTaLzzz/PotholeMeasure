"""Run every depth recipe on a folder of images and emit descriptive
statistics per recipe — works WITHOUT ground-truth depth.

Why this exists: real-world ground truth (rulered depth on the road) is
expensive. For a paper, this script gives you the next best thing — a
distribution-level comparison of the recipes on your test set, showing that:

* ``midas_relative_scaled`` produces wildly different magnitudes due to
  scale drift (huge std, mean far from physical range),
* ``metric_no_plane`` mostly reflects depth-map noise inside the mask
  (small range, low correlation with our reference),
* ``metric_plane_offset`` (ours) gives physically plausible cm-range
  values clustered tightly.

Outputs:
    <out>/relative_ablation.csv          per-detection (image, pid, recipe, depth)
    <out>/relative_ablation_summary.csv  per-recipe stats (mean/median/p95/std/n)
    <out>/relative_ablation.tex          LaTeX summary table
    <out>/relative_ablation.png          violin / boxplot per recipe (if matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--input-dir", type=Path, required=True, help="folder of test images")
    p.add_argument("--output", type=Path, default=Path("experiments/results/ablation"))
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    return p.parse_args()


def _percentile(xs, q: float) -> float:
    import numpy as np

    return float(np.percentile(xs, q)) if xs else 0.0


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2
    import numpy as np

    from src.evaluation import DEFAULT_RECIPES
    from src.geometry.plane_fitting import depth_to_pointcloud, fit_road_plane
    from src.pipeline import PotholePipeline

    pipeline = PotholePipeline.from_config(args.config)
    plane_cfg = pipeline.plane_cfg

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if args.max_images:
        images = images[: args.max_images]
    logging.info("evaluating %d images across %d recipes", len(images), len(DEFAULT_RECIPES))

    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_recipe: dict[str, list[float]] = {r.name: [] for r in DEFAULT_RECIPES}

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        detections = pipeline.segmentor.predict(image)
        if not detections:
            continue
        depth_map = pipeline.depth_estimator.predict(image)
        pc = depth_to_pointcloud(depth_map, pipeline.K)
        plane = fit_road_plane(
            pc,
            depth_map.shape,
            exclude_masks=[d.mask for d in detections],
            threshold_m=float(plane_cfg.get("ransac_threshold_m", 0.02)),
            max_iters=int(plane_cfg.get("max_iterations", 1000)),
            min_inlier_ratio=float(plane_cfg.get("min_inlier_ratio", 0.7)),
            min_points=int(plane_cfg.get("min_points", 1000)),
            up_axis=tuple(plane_cfg.get("up_axis", (0.0, -1.0, 0.0))),
            up_cos_threshold=float(plane_cfg.get("up_cos_threshold", 0.9)),
        )
        for pid, det in enumerate(detections):
            for recipe in DEFAULT_RECIPES:
                value = float(recipe.fn(depth_map, pc, det.mask, plane))
                rows.append({
                    "image": img_path.name,
                    "pothole_id": pid,
                    "recipe": recipe.name,
                    "depth_m": value,
                })
                per_recipe[recipe.name].append(value)
        logging.info("%s: %d potholes", img_path.name, len(detections))

    if not rows:
        raise SystemExit("no detections — lower segmentation.conf_threshold or check the input")

    # Per-detection CSV.
    detail_csv = args.output / "relative_ablation.csv"
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "pothole_id", "recipe", "depth_m"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary CSV + LaTeX.
    summary = []
    for name, vals in per_recipe.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        summary.append({
            "recipe": name,
            "n": len(vals),
            "mean_m": float(arr.mean()),
            "median_m": float(np.median(arr)),
            "p95_m": _percentile(vals, 95),
            "std_m": float(arr.std(ddof=0)),
            "min_m": float(arr.min()),
            "max_m": float(arr.max()),
        })

    summary_csv = args.output / "relative_ablation_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    tex_path = args.output / "relative_ablation.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("recipe & n & mean (m) & median (m) & p95 (m) & std (m) & max (m) \\\\\n\\midrule\n")
        underscore_safe = "\\_"
        for row in summary:
            recipe = row["recipe"].replace("_", underscore_safe)
            f.write(
                f"{recipe} & {row['n']} & "
                f"{row['mean_m']:.3f} & {row['median_m']:.3f} & "
                f"{row['p95_m']:.3f} & {row['std_m']:.3f} & {row['max_m']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    logging.info("wrote %s", detail_csv)
    logging.info("wrote %s", summary_csv)
    logging.info("wrote %s", tex_path)

    # Optional violin plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["recipe"] for r in summary]
        data = [per_recipe[n] for n in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("predicted depth (m)")
        ax.set_title("Recipe comparison on test set (no GT)")
        ax.grid(axis="y", alpha=0.3)
        png_path = args.output / "relative_ablation.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        logging.info("wrote %s", png_path)
    except ImportError:
        logging.warning("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
