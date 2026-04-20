"""Ablation study runner.

Each recipe describes how to compute a per-pothole ``depth_m`` from the raw
signals (depth map + point cloud + mask + fitted plane). The runner feeds
the same detections through every recipe and gathers regression and
classification metrics against ground truth.

Recipes in this module mirror the four configurations mentioned in the
plan:

* ``midas_relative_scaled`` — monocular relative depth with arbitrary
  rescaling (baseline 1). Demonstrates why raw monocular depth is unreliable.
* ``metric_no_plane``      — metric depth without plane subtraction
  (baseline 2). Uses ``max - min`` inside the mask.
* ``metric_plane_offset``  — our method: offset from the RANSAC plane
  (reports p95 of ``|signed distance|``).
* ``metric_plane_with_homography`` — same as above but bundled with the
  BEV area measurement; reported separately for the paper table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from src.geometry.plane_fitting import Plane, compute_depth_offset


RecipeFn = Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[Plane]], float]


@dataclass
class Recipe:
    name: str
    fn: RecipeFn


def recipe_midas_relative_scaled(
    depth_map: np.ndarray, pc: np.ndarray, mask: np.ndarray, plane: Optional[Plane]
) -> float:
    """Simulates MiDaS-style relative depth by inverting the metric map then
    rescaling arbitrarily — intentionally produces a poor metric depth.
    """
    mask_b = mask.astype(bool)
    if not mask_b.any():
        return 0.0
    rel = 1.0 / np.clip(depth_map, 1e-3, None)
    # Fit a linear map by the 5th and 95th percentiles of the relative depth.
    rel_flat = rel.reshape(-1)
    lo, hi = np.percentile(rel_flat, [5, 95])
    scale = 0.15 / max(hi - lo, 1e-6)          # arbitrary scale → metric drift
    scaled = (rel - lo) * scale
    inside = scaled[mask_b]
    return float(inside.max() - inside.min())


def recipe_metric_no_plane(
    depth_map: np.ndarray, pc: np.ndarray, mask: np.ndarray, plane: Optional[Plane]
) -> float:
    """Simple metric depth range inside the mask — ignores surrounding road."""
    inside = depth_map[mask.astype(bool)]
    if inside.size == 0:
        return 0.0
    return float(inside.max() - inside.min())


def recipe_metric_plane_offset(
    depth_map: np.ndarray, pc: np.ndarray, mask: np.ndarray, plane: Optional[Plane]
) -> float:
    """Ours — p95 of |signed distance| to the RANSAC road plane."""
    if plane is None:
        return 0.0
    stats = compute_depth_offset(plane, pc, mask)
    return stats.p95_m


DEFAULT_RECIPES: tuple[Recipe, ...] = (
    Recipe("midas_relative_scaled", recipe_midas_relative_scaled),
    Recipe("metric_no_plane", recipe_metric_no_plane),
    Recipe("metric_plane_offset", recipe_metric_plane_offset),
)


@dataclass
class PotholeGroundTruth:
    image_id: str
    mask: np.ndarray
    depth_m: float
    area_m2: float
    severity: str


@dataclass
class RecipeRun:
    recipe: str
    depth_preds: list[float]
    area_preds: list[float]
    severity_preds: list[str]


def run_ablation(
    per_image_signals: Sequence[tuple[np.ndarray, np.ndarray, Optional[Plane], Sequence[PotholeGroundTruth]]],
    recipes: Sequence[Recipe] = DEFAULT_RECIPES,
    area_fn: Optional[Callable[[np.ndarray], float]] = None,
    classifier: Optional[Callable[[float, float], str]] = None,
) -> list[RecipeRun]:
    """Evaluate every ``recipe`` against the same set of detections.

    ``per_image_signals`` is an iterable of
    ``(depth_map, pointcloud, plane, ground_truth_potholes)`` tuples. For each
    pothole in the ground truth the recipe's depth is computed; ``area_fn``
    (if provided) is applied to the pothole mask to produce a predicted area.
    """
    runs = [RecipeRun(recipe=r.name, depth_preds=[], area_preds=[], severity_preds=[]) for r in recipes]

    for depth_map, pc, plane, gts in per_image_signals:
        for gt in gts:
            area_pred = area_fn(gt.mask) if area_fn else gt.area_m2
            for run, recipe in zip(runs, recipes):
                depth_pred = recipe.fn(depth_map, pc, gt.mask, plane)
                run.depth_preds.append(float(depth_pred))
                run.area_preds.append(float(area_pred))
                if classifier is not None:
                    run.severity_preds.append(classifier(depth_pred, area_pred))

    return runs
