"""Road-plane fitting and per-pothole depth-offset estimation.

Central idea: monocular metric-depth models drift in absolute scale, so we
don't trust the raw metric values. Instead we fit a local plane to the road
surface with RANSAC over pixels that are NOT inside any pothole mask, then
measure each pothole's depth as the distance from that plane. Scale errors
common to the road and the pothole cancel out.

Coordinate convention
---------------------
Camera frame: +X right, +Y down, +Z forward (OpenCV convention).
The road plane therefore has a normal close to (0, -1, 0) (pointing up).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Plane:
    """Plane equation: ``normal · p + d = 0`` with ``normal`` oriented upward."""

    normal: np.ndarray   # (3,), unit length
    d: float
    inlier_ratio: float
    n_inliers: int

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.d


@dataclass
class DepthOffsetStats:
    max_m: float
    mean_m: float
    p95_m: float
    n_points: int


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a metric depth map into a ``(H*W, 3)`` point cloud.

    ``depth`` must be ``(H, W)`` float metres. ``K`` is the 3x3 intrinsic matrix
    matching that resolution.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    h, w = depth.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32, copy=False)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)


def _mask_union(masks: Iterable[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for m in masks:
        if m.shape != shape:
            raise ValueError(f"mask shape {m.shape} != target {shape}")
        out |= m.astype(bool)
    return out


def _ransac_plane(points: np.ndarray, threshold: float, max_iters: int, rng: np.random.Generator):
    n = len(points)
    if n < 3:
        return None, None
    best_plane = None
    best_inliers = None
    best_count = -1
    for _ in range(max_iters):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))
        dists = np.abs(points @ normal + d)
        inliers = dists < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_plane = (normal, d)
            best_inliers = inliers
    return best_plane, best_inliers


def _refine_plane(points: np.ndarray, inlier_mask: np.ndarray):
    inliers = points[inlier_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / float(np.linalg.norm(normal))
    d = -float(np.dot(normal, centroid))
    return normal, d


def fit_road_plane(
    pointcloud: np.ndarray,
    image_shape: tuple[int, int],
    exclude_masks: Iterable[np.ndarray] = (),
    *,
    threshold_m: float = 0.02,
    max_iters: int = 1000,
    min_inlier_ratio: float = 0.7,
    min_points: int = 1000,
    up_axis: Sequence[float] = (0.0, -1.0, 0.0),
    up_cos_threshold: float = 0.9,
    seed: int | None = 42,
) -> Plane | None:
    """RANSAC-fit the road plane over pixels outside all ``exclude_masks``.

    Returns ``None`` when any sanity check fails:
      * not enough valid points,
      * RANSAC degenerate,
      * inlier ratio below ``min_inlier_ratio``,
      * final normal not aligned with ``up_axis`` within ``up_cos_threshold``.
    """
    h, w = image_shape
    if pointcloud.shape != (h * w, 3):
        raise ValueError(
            f"pointcloud shape {pointcloud.shape} does not match image {(h, w)}"
        )

    exclude = _mask_union(list(exclude_masks), (h, w)).reshape(-1)
    valid = np.isfinite(pointcloud[:, 2]) & (pointcloud[:, 2] > 0)
    keep = valid & ~exclude
    candidates = pointcloud[keep]
    if len(candidates) < min_points:
        return None

    rng = np.random.default_rng(seed)
    coarse, inliers = _ransac_plane(candidates, threshold_m, max_iters, rng)
    if coarse is None or inliers is None:
        return None

    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / len(candidates)
    if inlier_ratio < min_inlier_ratio:
        return None

    normal, d = _refine_plane(candidates, inliers)

    up = np.asarray(up_axis, dtype=np.float32)
    up = up / float(np.linalg.norm(up))
    if float(np.dot(normal, up)) < 0:
        normal, d = -normal, -d
    if float(np.dot(normal, up)) < up_cos_threshold:
        return None

    return Plane(
        normal=normal.astype(np.float32),
        d=float(d),
        inlier_ratio=float(inlier_ratio),
        n_inliers=n_inliers,
    )


def compute_depth_offset(
    plane: Plane,
    pointcloud: np.ndarray,
    mask: np.ndarray,
) -> DepthOffsetStats:
    """Per-pothole depth statistics as ``|signed_distance|`` inside ``mask``."""
    mask_flat = mask.reshape(-1).astype(bool)
    points = pointcloud[mask_flat]
    points = points[np.isfinite(points[:, 2]) & (points[:, 2] > 0)]
    if len(points) == 0:
        return DepthOffsetStats(0.0, 0.0, 0.0, 0)
    abs_off = np.abs(plane.signed_distance(points))
    return DepthOffsetStats(
        max_m=float(abs_off.max()),
        mean_m=float(abs_off.mean()),
        p95_m=float(np.percentile(abs_off, 95)),
        n_points=int(len(points)),
    )
