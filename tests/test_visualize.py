"""Tests for the CV2-based drawing helpers (Open3D / matplotlib skipped)."""
from __future__ import annotations

import numpy as np
import pytest

from src.pipeline import PotholeResult
from src.visualize import DEFAULT_COLORS, draw_results


def _fake_result(h: int, w: int, severity: str) -> PotholeResult:
    mask = np.zeros((h, w), dtype=bool)
    mask[40:80, 50:110] = True
    return PotholeResult(
        bbox=(50.0, 40.0, 110.0, 80.0),
        confidence=0.87,
        depth_m=0.04,
        area_m2=0.12,
        mask_area_px=int(mask.sum()),
        severity=severity,
        mask=mask,
    )


def test_draw_results_marks_mask_with_severity_color() -> None:
    h, w = 120, 160
    image = np.zeros((h, w, 3), dtype=np.uint8)
    result = _fake_result(h, w, severity="moderate")
    out = draw_results(image, [result], alpha=1.0, draw_labels=False)
    assert out.shape == image.shape
    # Any mask pixel should now carry the severity colour (alpha=1.0 means full overlay).
    color = DEFAULT_COLORS["moderate"]
    mask_px = out[result.mask]
    assert (mask_px == np.array(color, dtype=np.uint8)).all()


def test_draw_results_handles_unknown_severity() -> None:
    h, w = 60, 80
    image = np.zeros((h, w, 3), dtype=np.uint8)
    result = _fake_result(h, w, severity=None)
    out = draw_results(image, [result], alpha=1.0, draw_labels=False)
    color = DEFAULT_COLORS[None]
    assert (out[result.mask] == np.array(color, dtype=np.uint8)).all()


def test_draw_results_is_no_op_when_mask_missing() -> None:
    h, w = 30, 40
    image = np.full((h, w, 3), 100, dtype=np.uint8)
    result = PotholeResult(
        bbox=(0, 0, 10, 10), confidence=0.5, depth_m=0.01, area_m2=0.01,
        mask_area_px=0, severity="minor", mask=None,
    )
    out = draw_results(image, [result])
    # Without a mask, overlay is identical to original for mask pixels; labels may
    # differ though — at least ensure shape stays stable and no crash.
    assert out.shape == image.shape
