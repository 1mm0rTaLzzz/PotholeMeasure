"""Visualisations for pothole detections, depth maps and 3D scenes.

All drawing helpers accept plain numpy arrays / dataclasses so they can be
called from notebooks as well as the CLI. Heavy libs (Open3D, matplotlib)
are imported lazily to keep this module cheap to import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


DEFAULT_COLORS = {
    "minor":    (0, 255, 0),
    "moderate": (0, 255, 255),
    "major":    (0, 128, 255),
    "critical": (0, 0, 255),
    None:       (200, 200, 200),
}


def _severity_color(severity, colors) -> tuple[int, int, int]:
    return tuple(colors.get(severity, colors[None]))


def draw_results(
    image: np.ndarray,
    potholes: Iterable,
    alpha: float = 0.5,
    colors: Optional[dict] = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Overlay pothole masks (colour-coded by severity) on top of ``image``.

    ``potholes`` is an iterable of ``PotholeResult``.
    """
    import cv2

    colors = {**DEFAULT_COLORS, **(colors or {})}
    overlay = image.copy()
    for p in potholes:
        if p.mask is None:
            continue
        color = _severity_color(p.severity, colors)
        overlay[p.mask] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    if draw_labels:
        for p in potholes:
            color = _severity_color(p.severity, colors)
            x0, y0, x1, y1 = [int(round(v)) for v in p.bbox]
            cv2.rectangle(blended, (x0, y0), (x1, y1), color, 2)
            depth_str = f"{p.depth_m * 100:.1f} cm" if p.depth_m and p.depth_m > 1e-4 else "n/a"
            label_lines = [
                f"{p.severity or '?'}",
                f"d={depth_str}  a={p.area_m2:.2f} m^2",
                f"conf={p.confidence:.2f}",
            ]
            _draw_label_block(blended, label_lines, anchor=(x0, max(0, y0 - 4)), color=color)

    return blended


def _draw_label_block(image, lines, anchor, color, scale=0.45, pad=3) -> None:
    import cv2

    x, y = anchor
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for t in lines]
    box_w = max(w for w, _ in sizes) + 2 * pad
    line_h = max(h for _, h in sizes) + 2
    box_h = line_h * len(lines) + 2 * pad
    y_box_top = max(0, y - box_h)
    cv2.rectangle(image, (x, y_box_top), (x + box_w, y_box_top + box_h), color, thickness=-1)
    for i, text in enumerate(lines):
        ty = y_box_top + pad + (i + 1) * line_h - 2
        cv2.putText(image, text, (x + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def plot_depth_comparison(
    image: np.ndarray,
    depth_map: np.ndarray,
    plane_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    dpi: int = 200,
):
    """Three-panel figure: RGB | depth | plane-inlier overlay. Returns the figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rgb = image[..., ::-1] if image.ndim == 3 else image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="viridis")
    axes[1].set_title("Depth (m)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if plane_mask is not None:
        overlay = rgb.copy() if rgb.ndim == 3 else np.stack([rgb] * 3, axis=-1)
        overlay[plane_mask.astype(bool)] = (overlay[plane_mask.astype(bool)] * 0.4 + np.array([0, 180, 0]) * 0.6).astype(overlay.dtype)
        axes[2].imshow(overlay)
        axes[2].set_title("Plane inliers")
    else:
        axes[2].imshow(depth_map, cmap="magma")
        axes[2].set_title("Depth (alt cmap)")
    axes[2].axis("off")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_3d_scene(
    pointcloud: np.ndarray,
    plane=None,
    potholes: Sequence = (),
    max_points: int = 200_000,
):
    """Interactive 3D view via Open3D. Returns a list of geometries if the
    caller wants to draw manually; otherwise opens a window.
    """
    import open3d as o3d

    if len(pointcloud) > max_points:
        idx = np.random.default_rng(0).choice(len(pointcloud), size=max_points, replace=False)
        pointcloud = pointcloud[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float64))

    geoms = [pcd]
    if plane is not None:
        mesh = o3d.geometry.TriangleMesh.create_box(10.0, 0.01, 10.0)
        mesh.translate(np.array([-5.0, -plane.d, -5.0]))
        mesh.paint_uniform_color([0.2, 0.8, 0.2])
        geoms.append(mesh)

    return geoms
