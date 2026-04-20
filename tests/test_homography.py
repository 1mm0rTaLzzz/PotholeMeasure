"""Tests for the ground-plane homography and area measurement."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry.homography import (
    compute_homography_from_calibration,
    intrinsics_from_fov,
    load_calibration,
    polygon_area_m2,
)


def _project_world_to_image(world_pts: np.ndarray, K: np.ndarray, H_m: float, pitch_deg: float) -> np.ndarray:
    theta = np.deg2rad(-pitch_deg)
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -s, H_m * c],
        [0.0, c, H_m * s],
    ], dtype=np.float64)
    H_w2i = K @ M
    homo = np.concatenate([world_pts, np.ones((len(world_pts), 1))], axis=1)
    img = homo @ H_w2i.T
    return img[:, :2] / img[:, 2:3]


def test_intrinsics_fov_matches_principal_point() -> None:
    K = intrinsics_from_fov(1280, 720, 60.0)
    assert K[0, 2] == pytest.approx(640.0)
    assert K[1, 2] == pytest.approx(360.0)
    # At 60° hfov on 1280 px, fx ≈ 1109.
    assert 1100.0 < K[0, 0] < 1120.0
    assert K[0, 0] == pytest.approx(K[1, 1])


def test_homography_inverts_forward_projection() -> None:
    K = intrinsics_from_fov(1280, 720, 60.0)
    H_img2world = compute_homography_from_calibration(K, camera_height_m=1.2, pitch_deg=-5.0)

    world_pts = np.array([[0.0, 5.0], [1.5, 8.0], [-1.5, 8.0], [2.0, 20.0]], dtype=np.float64)
    img_pts = _project_world_to_image(world_pts, K, H_m=1.2, pitch_deg=-5.0)

    img_h = np.concatenate([img_pts, np.ones((len(img_pts), 1))], axis=1)
    back = img_h @ H_img2world.T
    back = back[:, :2] / back[:, 2:3]
    assert np.allclose(back, world_pts, atol=1e-6)


def test_homography_forward_point_below_horizon() -> None:
    """A ground point 5 m in front of a downward-pitched camera projects below
    the principal point (v > cy)."""
    K = intrinsics_from_fov(1280, 720, 60.0)
    img = _project_world_to_image(
        np.array([[0.0, 5.0]]), K, H_m=1.2, pitch_deg=-5.0
    )
    assert img[0, 0] == pytest.approx(640.0, abs=0.5)
    assert img[0, 1] > 360.0


def test_polygon_area_recovers_known_rectangle() -> None:
    K = intrinsics_from_fov(1280, 720, 60.0)
    H_m, pitch = 1.2, -5.0
    H_img2world = compute_homography_from_calibration(K, H_m, pitch)

    # 0.5 m × 0.4 m rectangle on the ground, 6 m in front. Area = 0.20 m².
    corners_world = np.array([
        [-0.25, 5.8], [0.25, 5.8], [0.25, 6.2], [-0.25, 6.2]
    ], dtype=np.float64)
    corners_img = _project_world_to_image(corners_world, K, H_m, pitch)

    recovered = polygon_area_m2(corners_img, H_img2world)
    assert recovered == pytest.approx(0.20, rel=0.02)


def test_polygon_area_is_15pct_accurate_for_tilted_rect() -> None:
    # 1 m × 1 m square placed off-center, at ~8 m distance.
    K = intrinsics_from_fov(1280, 720, 60.0)
    H_m, pitch = 1.2, -5.0
    H_img2world = compute_homography_from_calibration(K, H_m, pitch)

    corners_world = np.array([
        [0.5, 7.5], [1.5, 7.5], [1.5, 8.5], [0.5, 8.5]
    ], dtype=np.float64)
    corners_img = _project_world_to_image(corners_world, K, H_m, pitch)

    recovered = polygon_area_m2(corners_img, H_img2world)
    assert 0.85 < recovered < 1.15    # within 15% of 1.0 m²


def test_polygon_area_rejects_degenerate() -> None:
    K = intrinsics_from_fov(1280, 720, 60.0)
    H_img2world = compute_homography_from_calibration(K, 1.2, -5.0)
    assert polygon_area_m2(np.array([[0, 0], [1, 1]]), H_img2world) == 0.0


def test_load_calibration_uses_fov_when_K_missing(tmp_path) -> None:
    yaml_path = tmp_path / "cam.yaml"
    yaml_path.write_text(
        "resolution:\n  width: 640\n  height: 480\n"
        "K: null\nhorizontal_fov_deg: 90.0\n"
        "mount:\n  height_m: 1.0\n  pitch_deg: 0.0\n  roll_deg: 0.0\n  yaw_deg: 0.0\n"
        "distortion:\n  model: none\n  coeffs: []\n"
    )
    K, cfg = load_calibration(yaml_path)
    assert K.shape == (3, 3)
    assert K[0, 2] == pytest.approx(320.0)
    assert cfg["mount"]["height_m"] == 1.0
