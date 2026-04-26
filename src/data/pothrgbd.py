"""Reader for the PothRGBD dataset (Kaggle / IEEE DataPort release).

PothRGBD ships RGB + depth pairs at 640x480 collected with an Intel
RealSense D415, with YOLOv8-seg style polygon labels exported via
Roboflow. Depth is a 16-bit PNG in millimetres (D415 default).

The Roboflow-exported folder commonly looks like one of these layouts;
this module auto-detects either::

    1) train/{images,labels,depth}/*  +  valid/...  +  test/...
    2) {images,labels,depth}/*        (flat, single split)
    3) <split>/{images,labels}/*  with depth alongside as `<stem>_depth.png`

YOLOv8-seg label lines: ``<cls> x1 y1 x2 y2 ...`` (coords normalised to
``[0, 1]``). The dataset is single-class (pothole), so we ignore ``cls``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


SUPPORTED_RGB_EXT = (".jpg", ".jpeg", ".png", ".bmp")
DEPTH_DIR_NAMES = ("depth", "Depth", "depth_maps", "depth_map", "depth_images", "depths")
SPLIT_NAMES = ("train", "valid", "val", "test")


@dataclass
class PothRGBDSample:
    rgb_path: Path
    depth_path: Path
    label_path: Path
    width: int = 0
    height: int = 0
    masks: list[np.ndarray] = field(default_factory=list, repr=False)


def _roboflow_base(stem: str) -> str:
    """Strip Roboflow's ``_color_png.rf.<hash>`` (or ``_color.rf.<hash>``) suffix
    to recover the original capture timestamp / id used by the depth files.

    Example::

        20250227_135438_color_png.rf.984e9768... -> 20250227_135438

    Returns the input unchanged when the marker is not present.
    """
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    for suffix in ("_color_png", "_color", "_rgb", "_jpg"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _depth_candidates(stem: str) -> list[str]:
    """All filename stems that could refer to the same capture as ``stem``."""
    base = _roboflow_base(stem)
    out = [stem, base]
    for suf in ("_depth", "-depth", ".depth", "_d"):
        out.append(f"{stem}{suf}")
        out.append(f"{base}{suf}")
    # Dedup while preserving order.
    seen: set[str] = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def _find_depth_for(rgb_path: Path) -> Path | None:
    """Locate the depth image paired with ``rgb_path`` under common Roboflow /
    PothRGBD conventions."""
    parent = rgb_path.parent
    candidates = _depth_candidates(rgb_path.stem)
    extensions = (".npy", ".png", ".tif", ".exr")

    # 1) Sibling folder named depth/depths/Depth/...
    for sibling in DEPTH_DIR_NAMES:
        sib_dir = parent.parent / sibling
        if not sib_dir.is_dir():
            continue
        for cand_stem in candidates:
            for ext in extensions:
                p = sib_dir / f"{cand_stem}{ext}"
                if p.exists():
                    return p

    # 2) Same folder, suffix-based naming (e.g. <stem>_depth.png).
    for cand_stem in candidates:
        if cand_stem == rgb_path.stem:
            continue
        for ext in extensions:
            p = parent / f"{cand_stem}{ext}"
            if p.exists():
                return p

    # 3) Same stem, different extension: <stem>.png vs <stem>.jpg.
    if rgb_path.suffix.lower() != ".png":
        p = parent / f"{rgb_path.stem}.png"
        if p.exists() and p != rgb_path:
            return p
    return None


def _find_label_for(rgb_path: Path) -> Path | None:
    stem = rgb_path.stem
    parent = rgb_path.parent
    # Roboflow standard: labels/ sibling.
    for label_dir in ("labels", "Labels", "annotations"):
        candidate = parent.parent / label_dir / f"{stem}.txt"
        if candidate.exists():
            return candidate
    # Fallback: alongside the image.
    candidate = parent / f"{stem}.txt"
    return candidate if candidate.exists() else None


def parse_yolo_seg_label(path: Path, width: int, height: int) -> list[np.ndarray]:
    """YOLO-seg ``<cls> x1 y1 x2 y2 ...`` -> list of bool ``(H, W)`` masks."""
    import cv2

    masks: list[np.ndarray] = []
    for raw in path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:                  # cls + at least 3 (x, y) pairs
            continue
        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        pts = np.array(
            [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        mask_u8 = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_u8, [pts], 1)
        if mask_u8.any():
            masks.append(mask_u8.astype(bool))
    return masks


def load_depth_metres(depth_path: Path, scale: float | None = None) -> np.ndarray:
    """Load a depth image and return metres as ``float32``.

    ``scale`` is the multiplier from on-disk units to metres. When ``None``,
    units are auto-detected:

    * ``.npy`` with integer dtype  -> assume mm, scale = 0.001
    * ``.npy`` with float dtype    -> assume metres already, scale = 1.0
    * ``.png`` / ``.tif`` (16-bit) -> D415 native mm, scale = 0.001

    Pass ``scale`` explicitly to override (e.g. when units are cm or
    deci-millimetres).
    """
    if depth_path.suffix.lower() == ".npy":
        arr = np.load(depth_path)
        if scale is None:
            scale = 0.001 if np.issubdtype(arr.dtype, np.integer) else 1.0
        return (arr.astype(np.float32) * float(scale))

    import cv2

    raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not decode depth image {depth_path}")
    if scale is None:
        scale = 0.001
    return raw.astype(np.float32) * float(scale)


def discover_pothrgbd(root: Path) -> dict[str, list[PothRGBDSample]]:
    """Walk the dataset root, group samples by split, and pair RGB / depth /
    YOLO-seg label paths. Returns ``{split: [PothRGBDSample, ...]}``."""
    logger = logging.getLogger(__name__)
    out: dict[str, list[PothRGBDSample]] = {}

    # Detect split folders. If none of train/valid/test exist, treat the
    # whole root as a single 'all' split.
    split_dirs: list[tuple[str, Path]] = []
    for name in SPLIT_NAMES:
        cand = root / name
        if cand.is_dir():
            split_dirs.append((name, cand))
    if not split_dirs:
        split_dirs = [("all", root)]

    for split, sd in split_dirs:
        # Look for an `images/` subdir, otherwise scan the whole split dir.
        images_dir = sd / "images" if (sd / "images").is_dir() else sd
        rgb_paths = sorted(
            p for p in images_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_RGB_EXT
            and "depth" not in p.parent.name.lower()
            and "label" not in p.parent.name.lower()
        )
        samples: list[PothRGBDSample] = []
        for rgb in rgb_paths:
            depth = _find_depth_for(rgb)
            if depth is None:
                continue
            label = _find_label_for(rgb)
            if label is None:
                continue
            samples.append(PothRGBDSample(rgb_path=rgb, depth_path=depth, label_path=label))
        out[split] = samples
        logger.info(
            "PothRGBD split %s: %d/%d images had a paired depth+label",
            split, len(samples), len(rgb_paths),
        )
    return out


def load_sample(sample: PothRGBDSample) -> PothRGBDSample:
    """Populate ``masks`` / ``width`` / ``height`` for a sample."""
    import cv2

    img = cv2.imread(str(sample.rgb_path))
    if img is None:
        raise FileNotFoundError(sample.rgb_path)
    h, w = img.shape[:2]
    masks = parse_yolo_seg_label(sample.label_path, w, h)
    sample.width = w
    sample.height = h
    sample.masks = masks
    return sample


def iter_loaded(samples: Iterable[PothRGBDSample]) -> Iterable[PothRGBDSample]:
    for s in samples:
        try:
            yield load_sample(s)
        except (FileNotFoundError, ValueError) as e:
            logging.warning("skipping %s: %s", s.rgb_path, e)
