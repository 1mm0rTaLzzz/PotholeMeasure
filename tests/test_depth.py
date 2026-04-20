"""Tests for the depth estimator pure-logic helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models.depth import DepthConfig, clip_depth, to_pil


def test_to_pil_accepts_pil_image() -> None:
    img = Image.new("RGB", (32, 16), color=(10, 20, 30))
    out = to_pil(img)
    assert isinstance(out, Image.Image)
    assert out.size == (32, 16)
    assert out.mode == "RGB"


def test_to_pil_reads_file(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 4), color=(1, 2, 3)).save(p)
    out = to_pil(p)
    assert out.size == (8, 4)


def test_to_pil_converts_bgr_ndarray_to_rgb() -> None:
    # Row of (B, G, R) pixels = (255, 0, 0). After BGR→RGB flip, R-channel should be 255.
    arr = np.zeros((2, 3, 3), dtype=np.uint8)
    arr[..., 0] = 255  # blue in BGR
    pil = to_pil(arr)
    rgb = np.asarray(pil)
    assert rgb.shape == (2, 3, 3)
    assert (rgb[..., 0] == 0).all()        # R
    assert (rgb[..., 2] == 255).all()      # B


def test_to_pil_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        to_pil(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_pil(np.zeros((10, 10, 4), dtype=np.uint8))


def test_to_pil_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        to_pil(1234)  # type: ignore[arg-type]


def test_clip_depth_bounds_and_dtype() -> None:
    raw = np.array([[-1.0, 0.0, 25.0, 200.0]])
    out = clip_depth(raw, min_m=0.5, max_m=80.0)
    assert out.dtype == np.float32
    assert out.min() == 0.5
    assert out.max() == 80.0
    assert out[0, 2] == pytest.approx(25.0)


def test_depth_config_defaults_match_yaml_model() -> None:
    cfg = DepthConfig()
    assert "Metric-Outdoor-Large" in cfg.model_id
    assert cfg.min_depth_m < cfg.max_depth_m
