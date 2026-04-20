"""Tests for metrics and the ablation runner."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    DEFAULT_RECIPES,
    PotholeGroundTruth,
    recipe_metric_no_plane,
    recipe_metric_plane_offset,
    run_ablation,
)
from src.geometry.metrics import classification_metrics, regression_metrics
from src.geometry.plane_fitting import Plane


def test_regression_metrics_basic() -> None:
    m = regression_metrics(preds=[0.05, 0.07, 0.10], gts=[0.04, 0.08, 0.10])
    assert m.n == 3
    assert m.mae == pytest.approx((0.01 + 0.01 + 0.0) / 3)
    assert m.rmse == pytest.approx(((0.01 ** 2 + 0.01 ** 2) / 3) ** 0.5)


def test_regression_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1, 2, 3])


def test_regression_metrics_ignores_zero_gt_in_rel_err() -> None:
    m = regression_metrics(preds=[0.1, 0.2], gts=[0.0, 0.2])
    assert m.mean_abs_rel_err == pytest.approx(0.0)


def test_classification_metrics_perfect() -> None:
    m = classification_metrics(["a", "b", "a"], ["a", "b", "a"])
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.per_class_f1 == {"a": 1.0, "b": 1.0}


def test_classification_metrics_all_wrong() -> None:
    m = classification_metrics(["a", "b"], ["b", "a"])
    assert m.accuracy == 0.0
    assert m.macro_f1 == 0.0


def test_classification_metrics_partial() -> None:
    m = classification_metrics(
        ["minor", "minor", "major", "critical"],
        ["minor", "major", "major", "critical"],
    )
    assert m.accuracy == pytest.approx(0.75)
    assert set(m.per_class_f1) == {"minor", "major", "critical"}


def test_recipe_metric_no_plane_is_max_minus_min() -> None:
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[5:7, 5:7] = 0.15
    mask = np.zeros_like(depth, dtype=bool)
    mask[4:8, 4:8] = True
    val = recipe_metric_no_plane(depth, np.zeros((100, 3)), mask, plane=None)
    assert val == pytest.approx(0.15)


def test_recipe_plane_offset_matches_synthetic_pit() -> None:
    # Build a simple flat cloud on plane y=1.2; shift a small region 5cm.
    h, w = 40, 50
    xs = np.linspace(-2, 2, w, dtype=np.float32)
    zs = np.linspace(2, 10, h, dtype=np.float32)
    gx, gz = np.meshgrid(xs, zs)
    gy = np.full_like(gx, 1.2, dtype=np.float32)
    pc = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    mask = np.zeros((h, w), dtype=bool)
    mask[15:20, 20:30] = True
    pc_img = pc.reshape(h, w, 3).copy()
    pc_img[mask] += np.array([0.0, 0.05, 0.0], dtype=np.float32)
    pc = pc_img.reshape(-1, 3)
    plane = Plane(normal=np.array([0, -1, 0], dtype=np.float32), d=1.2, inlier_ratio=1.0, n_inliers=1)
    val = recipe_metric_plane_offset(np.zeros((h, w), dtype=np.float32), pc, mask, plane)
    assert val == pytest.approx(0.05, abs=1e-3)


def test_run_ablation_emits_per_recipe_results() -> None:
    h, w = 20, 20
    pc = np.stack([
        np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32), (h, w)),
        np.full((h, w), 1.2, dtype=np.float32),
        np.broadcast_to(np.linspace(2, 5, h, dtype=np.float32)[:, None], (h, w)),
    ], axis=-1).reshape(-1, 3)
    plane = Plane(normal=np.array([0, -1, 0], dtype=np.float32), d=1.2, inlier_ratio=1.0, n_inliers=1)
    depth_map = np.full((h, w), 3.0, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[8:12, 8:12] = True
    pc_img = pc.reshape(h, w, 3).copy()
    pc_img[mask] += np.array([0.0, 0.08, 0.0], dtype=np.float32)
    pc_shift = pc_img.reshape(-1, 3)
    gt = PotholeGroundTruth(image_id="x", mask=mask, depth_m=0.08, area_m2=0.1, severity="major")

    runs = run_ablation(
        [(depth_map, pc_shift, plane, [gt])],
        recipes=DEFAULT_RECIPES,
        area_fn=lambda m: 0.1,
    )
    assert [r.recipe for r in runs] == [r.name for r in DEFAULT_RECIPES]
    for run in runs:
        assert len(run.depth_preds) == 1
        assert len(run.area_preds) == 1
    # Plane-offset recipe should recover ~0.08 m; mean-depth (zero variation) ~0.
    offset_run = next(r for r in runs if r.recipe == "metric_plane_offset")
    assert offset_run.depth_preds[0] == pytest.approx(0.08, abs=0.01)
    mean_run = next(r for r in runs if r.recipe == "metric_no_plane")
    assert mean_run.depth_preds[0] == pytest.approx(0.0, abs=1e-5)
