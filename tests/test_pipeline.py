"""Integration tests for the PotholePipeline wiring."""
from __future__ import annotations

import numpy as np
import pytest

from src.geometry.homography import compute_homography_from_calibration, intrinsics_from_fov
from src.models.segmentation import PotholeDetection
from src.pipeline import PotholePipeline


class _FakeSegmentor:
    def __init__(self, detections):
        self._detections = detections

    def predict(self, image):
        return list(self._detections)


class _FakeDepth:
    def __init__(self, depth_map):
        self._depth = depth_map

    def predict(self, image):
        return self._depth


def _flat_depth(K: np.ndarray, h: int, w: int, camera_height: float = 1.2, pitch_deg: float = -5.0) -> np.ndarray:
    """Synthesise a depth map consistent with a flat road at given mounting."""
    theta = np.deg2rad(-pitch_deg)
    fy = float(K[1, 1])
    cy = float(K[1, 2])
    v = np.arange(h, dtype=np.float32)
    yp = (v - cy) / fy
    # Road equation in camera frame: Y_c = H·cos(θ) - Z·sin(θ) ⇒ Z = (H·cos θ) / (yp·cos θ + sin θ)
    # From world math we used: Y_c = -Y_world·sin θ - Z_world·cos θ + H·cos θ, Z_c = Y_world·cos θ - Z_world·sin θ + H·sin θ.
    # For Z_world = 0: Y_c = -Y_w·sin θ + H·cos θ ; Z_c = Y_w·cos θ + H·sin θ.
    # Image ray: yp = Y_c / Z_c. Solve Y_w: yp (Y_w cos θ + H sin θ) = -Y_w sin θ + H cos θ
    #           → Y_w (yp cos θ + sin θ) = H (cos θ - yp sin θ)
    #           → Y_w = H (cos θ - yp sin θ) / (yp cos θ + sin θ)
    # Then Z_c = Y_w cos θ + H sin θ.
    Yw = camera_height * (np.cos(theta) - yp * np.sin(theta)) / (yp * np.cos(theta) + np.sin(theta))
    Zc = Yw * np.cos(theta) + camera_height * np.sin(theta)
    Zc = np.where(Zc > 0, Zc, 100.0)        # above-horizon rows → far value (clipped by pipeline)
    return np.broadcast_to(Zc[:, None], (h, w)).astype(np.float32).copy()


def test_pipeline_returns_per_pothole_measurements() -> None:
    h, w = 240, 320
    K = intrinsics_from_fov(w, h, 60.0)
    H = compute_homography_from_calibration(K, camera_height_m=1.2, pitch_deg=-5.0)
    depth = _flat_depth(K, h, w)

    # Build a pothole mask well below the horizon.
    mask = np.zeros((h, w), dtype=bool)
    mask[170:190, 130:180] = True
    # Depress the mask region ~6 cm along the "up" axis relative to the road surface.
    # Since synthetic depth is purely planar, we adjust the depth value inside the
    # mask so the back-projected points sit 0.06 m above the road (which after
    # |signed_distance| gives ~6 cm depth).
    depth[mask] += 0.06 * np.sqrt(
        1 + ((np.arange(w) - K[0, 2]) / K[0, 0]) ** 2 + ((np.arange(h)[:, None] - K[1, 2]) / K[1, 1]) ** 2
    )[mask].astype(np.float32)

    detection = PotholeDetection(mask=mask, bbox=(130.0, 170.0, 180.0, 190.0), score=0.88)
    pipeline = PotholePipeline(
        segmentor=_FakeSegmentor([detection]),
        depth_estimator=_FakeDepth(depth),
        K=K,
        H_img2world=H,
        plane_cfg={"min_points": 500, "min_inlier_ratio": 0.5, "ransac_threshold_m": 0.02},
    )

    frame = pipeline.process(np.zeros((h, w, 3), dtype=np.uint8))
    assert frame.plane is not None
    assert frame.plane_fit_failed is False
    assert len(frame.potholes) == 1
    pot = frame.potholes[0]
    assert 0.01 < pot.depth_m < 0.20               # non-trivial depth in metres
    assert pot.area_m2 > 0.0
    assert pot.mask_area_px == int(mask.sum())
    assert pot.confidence == pytest.approx(0.88)


def test_pipeline_reports_plane_failure_gracefully() -> None:
    h, w = 60, 80
    K = intrinsics_from_fov(w, h, 60.0)
    H = compute_homography_from_calibration(K, camera_height_m=1.2, pitch_deg=-5.0)
    depth = np.full((h, w), 10.0, dtype=np.float32)
    # No mask → plane cannot be rejected by min-inlier ratio, but we force
    # min_points higher than the image to trigger failure.
    pipeline = PotholePipeline(
        segmentor=_FakeSegmentor([]),
        depth_estimator=_FakeDepth(depth),
        K=K,
        H_img2world=H,
        plane_cfg={"min_points": 1_000_000},
    )
    frame = pipeline.process(np.zeros((h, w, 3), dtype=np.uint8))
    assert frame.plane is None
    assert frame.plane_fit_failed is True
    assert frame.potholes == []


def test_frame_result_serialises_to_json() -> None:
    h, w = 120, 160
    K = intrinsics_from_fov(w, h, 60.0)
    H = compute_homography_from_calibration(K, camera_height_m=1.2, pitch_deg=-5.0)
    depth = _flat_depth(K, h, w)
    mask = np.zeros((h, w), dtype=bool)
    mask[80:95, 60:90] = True
    pipeline = PotholePipeline(
        segmentor=_FakeSegmentor([PotholeDetection(mask=mask, bbox=(60, 80, 90, 95), score=0.9)]),
        depth_estimator=_FakeDepth(depth),
        K=K,
        H_img2world=H,
        plane_cfg={"min_points": 500, "min_inlier_ratio": 0.5},
    )
    import json

    payload = pipeline.process(np.zeros((h, w, 3), dtype=np.uint8)).to_json_dict()
    dumped = json.dumps(payload)
    assert "potholes" in dumped
    assert "plane" in dumped
