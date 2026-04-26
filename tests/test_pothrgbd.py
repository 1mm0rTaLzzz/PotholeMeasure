"""Tests for the PothRGBD reader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.pothrgbd import (
    discover_pothrgbd,
    load_depth_metres,
    parse_yolo_seg_label,
)


def _write_dummy(rgb_dir: Path, depth_dir: Path, label_dir: Path, stem: str, w: int = 64, h: int = 48):
    import cv2

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rgb_dir / f"{stem}.jpg"), np.zeros((h, w, 3), dtype=np.uint8))
    # depth in mm (D415 native) — uniform 1500 mm = 1.5 m.
    depth_mm = np.full((h, w), 1500, dtype=np.uint16)
    cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
    # one normalised polygon roughly in the middle.
    polygon = "0 0.30 0.30 0.70 0.30 0.70 0.70 0.30 0.70"
    (label_dir / f"{stem}.txt").write_text(polygon)


def test_yolo_seg_label_parsed_to_mask(tmp_path: Path) -> None:
    label = tmp_path / "img.txt"
    label.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    masks = parse_yolo_seg_label(label, width=100, height=100)
    assert len(masks) == 1
    m = masks[0]
    assert m.dtype == bool
    assert m[50, 50] is np.True_ or m[50, 50] == True   # noqa: E712
    assert not m[5, 5]


def test_load_depth_metres_converts_units(tmp_path: Path) -> None:
    import cv2

    arr = np.full((10, 10), 2500, dtype=np.uint16)            # 2500 mm
    cv2.imwrite(str(tmp_path / "d.png"), arr)
    d = load_depth_metres(tmp_path / "d.png", scale=0.001)
    assert d.dtype == np.float32
    assert abs(float(d.mean()) - 2.5) < 1e-3


def test_discover_layout_split_dirs(tmp_path: Path) -> None:
    # Roboflow-ish layout: train/{images,labels,depth} + valid/...
    for split in ("train", "valid"):
        _write_dummy(
            rgb_dir=tmp_path / split / "images",
            depth_dir=tmp_path / split / "depth",
            label_dir=tmp_path / split / "labels",
            stem=f"{split}_a",
        )
    found = discover_pothrgbd(tmp_path)
    assert sorted(found) == ["train", "valid"]
    assert len(found["train"]) == 1
    assert found["valid"][0].rgb_path.name == "valid_a.jpg"
    assert found["valid"][0].depth_path.exists()
    assert found["valid"][0].label_path.exists()


def test_roboflow_naming_pairs_depth_by_timestamp(tmp_path: Path) -> None:
    """PothRGBD-on-Kaggle pattern:
        images/<TS>_color_png.rf.<HASH>.jpg
        depths/<TS>_depth.npy
        labels/<TS>_color_png.rf.<HASH>.txt
    """
    import cv2

    rgb_dir = tmp_path / "images"
    depth_dir = tmp_path / "depths"
    label_dir = tmp_path / "labels"
    for d in (rgb_dir, depth_dir, label_dir):
        d.mkdir(parents=True)

    ts = "20250227_135438"
    rgb_stem = f"{ts}_color_png.rf.984e9768"
    cv2.imwrite(str(rgb_dir / f"{rgb_stem}.jpg"), np.zeros((48, 64, 3), dtype=np.uint8))
    np.save(depth_dir / f"{ts}_depth.npy", np.full((48, 64), 1.5, dtype=np.float32))
    (label_dir / f"{rgb_stem}.txt").write_text("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")

    found = discover_pothrgbd(tmp_path)
    assert "all" in found
    assert len(found["all"]) == 1
    s = found["all"][0]
    assert s.depth_path.name == f"{ts}_depth.npy"
    # auto-detect: float .npy -> metres, no rescaling.
    d = load_depth_metres(s.depth_path)
    assert abs(float(d.mean()) - 1.5) < 1e-3


def test_discover_layout_flat(tmp_path: Path) -> None:
    _write_dummy(
        rgb_dir=tmp_path / "images",
        depth_dir=tmp_path / "depth",
        label_dir=tmp_path / "labels",
        stem="x",
    )
    found = discover_pothrgbd(tmp_path)
    assert "all" in found
    assert len(found["all"]) == 1
