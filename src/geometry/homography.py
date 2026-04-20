"""Ground-plane homography and per-mask area measurement.

We model the road in front of the vehicle as a flat plane. Given the camera
intrinsics and mounting geometry (height + pitch), we derive a closed-form
homography mapping image pixels to real-world metres on that plane. The
inverse map lets us project a pothole mask into a bird's-eye view and
compute its area via the shoelace formula.

Coordinate conventions
----------------------
Camera frame: OpenCV — +X right, +Y down, +Z forward.
World frame: +X right, +Y forward (into scene), +Z up.
Ground plane: Z = 0, with the camera directly above world origin at Z = H.

``pitch_deg`` in this module follows the dashcam convention: **negative =
camera tilted downward** (e.g. ``-5.0``). Zero means the optical axis is
horizontal. The internal math converts this to a positive downward angle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """Build a 3×3 intrinsic matrix assuming square pixels and zero skew."""
    fx = 0.5 * width / np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_calibration(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """Load ``camera.yaml``. Returns ``(K, parsed_dict)``."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    w = int(cfg["resolution"]["width"])
    h = int(cfg["resolution"]["height"])
    if cfg.get("K"):
        K = np.asarray(cfg["K"], dtype=np.float64)
    else:
        K = intrinsics_from_fov(w, h, float(cfg["horizontal_fov_deg"]))
    return K, cfg


def compute_homography_from_calibration(
    K: np.ndarray,
    camera_height_m: float,
    pitch_deg: float,
) -> np.ndarray:
    """Homography H mapping image pixels → world metres on the ground plane.

    For a world point ``(X, Y)`` (X right, Y forward, both in metres) on the
    ground, the forward homography H_w2i maps ``(X, Y, 1)`` to homogeneous
    image pixels. This function returns its inverse so callers can transform
    image polygons directly into BEV metres.
    """
    theta = np.deg2rad(-pitch_deg)       # dashcam pitch is negative-down → θ > 0
    c = np.cos(theta)
    s = np.sin(theta)
    # Columns 1, 2 of R_wc + translation vector.
    M = np.array([
        [1.0, 0.0,                  0.0],
        [0.0, -s,   camera_height_m * c],
        [0.0,  c,   camera_height_m * s],
    ], dtype=np.float64)
    H_w2i = np.asarray(K, dtype=np.float64) @ M
    return np.linalg.inv(H_w2i)


def compute_homography_from_lanes(
    image: "np.ndarray",  # type: ignore[name-defined]
    lane_width_m: float = 3.5,
) -> np.ndarray:
    """Fallback homography from detected lane markings.

    NOTE: a full lane-detection pipeline is out of scope for this phase. This
    stub raises ``NotImplementedError`` so production callers must configure
    camera.yaml instead. Left in place so the interface matches the plan.
    """
    raise NotImplementedError(
        "lane-based homography is not implemented; use camera.yaml calibration"
    )


def _project_points(points_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography to an ``(N, 2)`` array of pixel coords."""
    pts_h = np.concatenate([points_px, np.ones((len(points_px), 1))], axis=1)
    proj = pts_h @ H.T
    w = proj[:, 2:3]
    valid = np.abs(w) > 1e-9
    out = np.full((len(points_px), 2), np.nan, dtype=np.float64)
    out[valid[:, 0]] = proj[valid[:, 0], :2] / w[valid[:, 0]]
    return out


def _shoelace(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_to_area_m2(mask: np.ndarray, H_img2world: np.ndarray) -> float:
    """Area of a binary mask in world square metres via BEV projection."""
    import cv2

    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    total = 0.0
    for cnt in contours:
        pts_px = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts_px) < 3:
            continue
        pts_w = _project_points(pts_px, H_img2world)
        if np.isnan(pts_w).any():
            continue
        total += _shoelace(pts_w)
    return total


def polygon_area_m2(points_px: np.ndarray, H_img2world: np.ndarray) -> float:
    """Area of an arbitrary pixel polygon projected through ``H``."""
    points_px = np.asarray(points_px, dtype=np.float64)
    if len(points_px) < 3:
        return 0.0
    pts_w = _project_points(points_px, H_img2world)
    if np.isnan(pts_w).any():
        return 0.0
    return _shoelace(pts_w)
