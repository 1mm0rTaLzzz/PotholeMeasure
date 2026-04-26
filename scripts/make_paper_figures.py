"""Stitch paper-ready figures from a finished run_inference.py output.

Reads ``--inference-dir`` (the folder produced by run_inference.py with
``viz/`` + ``json/`` subfolders), groups detections by severity, and emits:

* ``severity_grid.jpg`` — 2x2 grid: minor / moderate / major / critical
  (best example by max(depth*area) per bucket).
* ``hero.jpg``         — single largest pothole detection (for teaser).
* ``stats.json``       — per-severity counts and mean depth/area.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SEVERITY_ORDER = ("minor", "moderate", "major", "critical")
SEVERITY_COLORS = {
    "minor":    "#2ecc71",
    "moderate": "#f1c40f",
    "major":    "#e67e22",
    "critical": "#e74c3c",
}


def _emit_area_distribution(by_sev: dict, stats: dict, out_dir: Path) -> None:
    """Histogram of area_m2 per severity bucket + LaTeX table of stats."""
    import numpy as np

    # LaTeX summary.
    tex = out_dir / "severity_area_stats.tex"
    with tex.open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("severity & n & mean (m^2) & median (m^2) & p95 (m^2) & std (m^2) & max (m^2) \\\\\n\\midrule\n")
        for sev in SEVERITY_ORDER:
            s = stats.get(sev) or {}
            if s.get("n", 0) == 0:
                f.write(f"{sev} & 0 & - & - & - & - & - \\\\\n")
                continue
            a = s["area_m2"]
            f.write(
                f"{sev} & {s['n']} & {a['mean']:.3f} & {a['median']:.3f} & "
                f"{a['p95']:.3f} & {a['std']:.3f} & {a['max']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for sev in SEVERITY_ORDER:
        entries = by_sev.get(sev, [])
        if not entries:
            continue
        areas = np.asarray([e[2].get("area_m2", 0.0) for e in entries], dtype=np.float64)
        # Clip the long tail so a few critical outliers don't flatten the histogram.
        clip_hi = float(np.quantile(areas, 0.99)) if len(areas) > 5 else float(areas.max())
        areas = areas[areas <= max(clip_hi, 1e-3)]
        if not areas.size:
            continue
        has_data = True
        ax.hist(
            areas, bins=30, alpha=0.55,
            label=f"{sev} (n={len(entries)})",
            color=SEVERITY_COLORS.get(sev, "#888"),
            edgecolor="black", linewidth=0.5,
        )
    if has_data:
        ax.set_xlabel("BEV area (m^2)")
        ax.set_ylabel("count")
        ax.set_title("Area distribution by severity bucket")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "area_distribution.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inference-dir", type=Path, required=True,
                   help="folder produced by scripts/run_inference.py (must contain viz/ + json/)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="where to write the figures (default: <inference-dir>/figures)")
    return p.parse_args()


def _score(p: dict) -> float:
    return float(p.get("depth_m") or 0.0) * float(p.get("area_m2") or 0.0) + 1e-3 * float(p.get("confidence") or 0.0)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import cv2
    import numpy as np

    out_dir = args.output_dir or (args.inference_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = args.inference_dir / "json"
    viz_dir = args.inference_dir / "viz"
    if not json_dir.is_dir() or not viz_dir.is_dir():
        raise SystemExit(f"missing json/ or viz/ inside {args.inference_dir}")

    # Group: severity → list[(score, image_stem, pothole_dict)].
    by_sev: dict[str, list[tuple[float, str, dict]]] = {s: [] for s in SEVERITY_ORDER}
    everyone: list[tuple[float, str, dict]] = []
    for jpath in sorted(json_dir.glob("*.json")):
        try:
            payload = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            continue
        for p in payload.get("potholes", []):
            sev = p.get("severity") or "minor"
            score = _score(p)
            entry = (score, jpath.stem, p)
            everyone.append(entry)
            if sev in by_sev:
                by_sev[sev].append(entry)

    if not everyone:
        raise SystemExit("no detections in json/ — nothing to figure")

    # Per-severity stats.
    stats = {}
    for sev, entries in by_sev.items():
        if not entries:
            stats[sev] = {"n": 0}
            continue
        depths = [e[2].get("depth_m", 0.0) for e in entries]
        areas = [e[2].get("area_m2", 0.0) for e in entries]
        a = np.asarray(areas, dtype=np.float64)
        d = np.asarray(depths, dtype=np.float64)
        stats[sev] = {
            "n": len(entries),
            "area_m2": {
                "mean":   float(a.mean()),
                "median": float(np.median(a)),
                "p95":    float(np.percentile(a, 95)),
                "std":    float(a.std(ddof=0)),
                "min":    float(a.min()),
                "max":    float(a.max()),
            },
            "depth_m": {
                "mean":   float(d.mean()),
                "median": float(np.median(d)),
                "p95":    float(np.percentile(d, 95)),
            },
        }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logging.info("severity counts: %s", {k: v["n"] for k, v in stats.items()})

    # Area distribution per severity bucket — histogram + LaTeX table.
    _emit_area_distribution(by_sev, stats, out_dir)

    # Hero: detection with largest score across the whole run.
    everyone.sort(key=lambda x: -x[0])
    hero_score, hero_stem, _ = everyone[0]
    hero_path = viz_dir / f"{hero_stem}.jpg"
    if hero_path.exists():
        hero = cv2.imread(str(hero_path))
        cv2.imwrite(str(out_dir / "hero.jpg"), hero)
        logging.info("hero from %s (score=%.4f)", hero_stem, hero_score)

    # 2x2 grid of severity examples.
    panels = []
    for sev in SEVERITY_ORDER:
        if not by_sev[sev]:
            panels.append(None)
            continue
        by_sev[sev].sort(key=lambda x: -x[0])
        _, stem, _ = by_sev[sev][0]
        path = viz_dir / f"{stem}.jpg"
        if path.exists():
            panels.append((sev, cv2.imread(str(path))))
        else:
            panels.append(None)

    valid = [p for p in panels if p is not None]
    if not valid:
        logging.warning("no severity panels found")
        return

    # Pad to common shape (smallest of valid, then resize all).
    h_min = min(img.shape[0] for _, img in valid)
    w_min = min(img.shape[1] for _, img in valid)

    def _prep(img: "np.ndarray", label: str) -> "np.ndarray":
        resized = cv2.resize(img, (w_min, h_min))
        cv2.rectangle(resized, (0, 0), (260, 36), (0, 0, 0), -1)
        cv2.putText(resized, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return resized

    blank = np.full((h_min, w_min, 3), 32, dtype=np.uint8)
    grid_imgs: list = []
    for slot, sev in zip(panels, SEVERITY_ORDER):
        if slot is None:
            cv2.putText(blank.copy(), f"{sev}: n=0", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            grid_imgs.append(blank.copy())
        else:
            _, img = slot
            grid_imgs.append(_prep(img, f"{sev} (n={stats[sev]['n']})"))

    top = np.hstack(grid_imgs[:2])
    bot = np.hstack(grid_imgs[2:])
    grid = np.vstack([top, bot])
    cv2.imwrite(str(out_dir / "severity_grid.jpg"), grid)
    logging.info("wrote %s", out_dir / "severity_grid.jpg")


if __name__ == "__main__":
    main()
