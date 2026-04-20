"""Synthetic validation of the RANSAC road-plane + depth-offset pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry.plane_fitting import (
    Plane,
    compute_depth_offset,
    depth_to_pointcloud,
    fit_road_plane,
)


def _flat_road_pointcloud(h: int, w: int, height_m: float = 1.2) -> np.ndarray:
    """Build a (H*W, 3) point cloud for a flat road at Y = +height_m (OpenCV down-axis)."""
    xs = np.linspace(-3.0, 3.0, w, dtype=np.float32)
    zs = np.linspace(2.0, 20.0, h, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = np.full_like(grid_x, height_m, dtype=np.float32)
    return np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)


def test_depth_to_pointcloud_back_projects_principal_point() -> None:
    depth = np.full((4, 6), 10.0, dtype=np.float32)
    K = np.array([[100, 0, 3], [0, 100, 2], [0, 0, 1]], dtype=np.float32)
    pc = depth_to_pointcloud(depth, K)
    idx = 2 * 6 + 3          # pixel (u=3, v=2) = principal point
    assert pc[idx] == pytest.approx([0.0, 0.0, 10.0], abs=1e-5)


def test_depth_to_pointcloud_scales_with_focal_length() -> None:
    depth = np.full((2, 2), 5.0, dtype=np.float32)
    K = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 1]], dtype=np.float32)
    pc = depth_to_pointcloud(depth, K)
    # Pixel (1, 1) with K=(200, 0, 0) at depth 5 → X = 1/200 * 5 = 0.025.
    assert pc[3, 0] == pytest.approx(0.025)
    assert pc[3, 1] == pytest.approx(0.025)
    assert pc[3, 2] == pytest.approx(5.0)


def test_depth_to_pointcloud_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        depth_to_pointcloud(np.zeros((2, 2, 2), dtype=np.float32), np.eye(3, dtype=np.float32))


def test_fit_road_plane_recovers_flat_road() -> None:
    h, w = 60, 80
    pc = _flat_road_pointcloud(h, w, height_m=1.2)
    # Add mild gaussian noise so the test exercises RANSAC (<< 2 cm threshold).
    rng = np.random.default_rng(0)
    pc = pc + rng.normal(scale=0.005, size=pc.shape).astype(np.float32)
    plane = fit_road_plane(pc, (h, w))
    assert plane is not None
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    assert float(np.dot(plane.normal, up)) > 0.99
    assert plane.inlier_ratio > 0.9
    # d should equal +1.2 (since normal points up, and plane is at Y = 1.2).
    assert plane.d == pytest.approx(1.2, abs=0.02)


def test_fit_road_plane_rejects_non_up_facing_surface() -> None:
    # Construct a plane whose normal is (0, 0, 1): a vertical wall facing the camera.
    h, w = 50, 50
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pc = np.stack([gx, gy, np.full_like(gx, 5.0)], axis=-1).reshape(-1, 3)
    plane = fit_road_plane(pc, (h, w))
    assert plane is None, "vertical wall must be rejected by up-axis check"


def test_fit_road_plane_returns_none_when_not_enough_points() -> None:
    h, w = 10, 10
    pc = _flat_road_pointcloud(h, w)
    plane = fit_road_plane(pc, (h, w), min_points=500)
    assert plane is None


def test_fit_road_plane_excludes_mask() -> None:
    h, w = 60, 80
    pc = _flat_road_pointcloud(h, w, height_m=1.2).reshape(h, w, 3).copy()
    mask = np.zeros((h, w), dtype=bool)
    mask[25:35, 35:50] = True
    # Deform the pothole region so it is NOT on the plane.
    pc[mask] += np.array([0.0, 0.08, 0.0], dtype=np.float32)
    pc_flat = pc.reshape(-1, 3)
    # Without exclude → inlier ratio drops. With exclude → full road stays clean.
    plane = fit_road_plane(pc_flat, (h, w), exclude_masks=[mask])
    assert plane is not None
    assert plane.inlier_ratio > 0.98


def test_compute_depth_offset_matches_synthetic_pit() -> None:
    h, w = 60, 80
    pit_depth_m = 0.08       # 8 cm
    pc = _flat_road_pointcloud(h, w, height_m=1.2).reshape(h, w, 3).copy()
    mask = np.zeros((h, w), dtype=bool)
    mask[25:35, 35:50] = True
    # Move pothole points along the "up" axis direction away from camera → deeper pit.
    pc[mask] += np.array([0.0, pit_depth_m, 0.0], dtype=np.float32)
    pc_flat = pc.reshape(-1, 3)

    plane = fit_road_plane(pc_flat, (h, w), exclude_masks=[mask])
    assert plane is not None
    stats = compute_depth_offset(plane, pc_flat, mask)
    assert stats.n_points == int(mask.sum())
    assert stats.p95_m == pytest.approx(pit_depth_m, abs=0.005)
    assert stats.max_m >= stats.p95_m
    assert stats.mean_m == pytest.approx(pit_depth_m, abs=0.005)


def test_compute_depth_offset_empty_mask() -> None:
    pc = np.zeros((10, 3), dtype=np.float32)
    pc[:, 2] = 5.0
    plane = Plane(normal=np.array([0, -1, 0], dtype=np.float32), d=1.2, inlier_ratio=1.0, n_inliers=10)
    stats = compute_depth_offset(plane, pc, mask=np.zeros((2, 5), dtype=bool))
    assert stats == type(stats)(0.0, 0.0, 0.0, 0)


def test_plane_signed_distance_is_zero_on_plane() -> None:
    plane = Plane(normal=np.array([0, -1, 0], dtype=np.float32), d=1.2, inlier_ratio=1.0, n_inliers=1)
    on_plane = np.array([[0.5, 1.2, 5.0], [-0.3, 1.2, 10.0]], dtype=np.float32)
    dists = plane.signed_distance(on_plane)
    assert np.allclose(dists, 0.0, atol=1e-6)
