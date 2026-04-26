"""Severity classification aligned with GOST R 50597-2017 thresholds.

A pothole is classified at the *worst* severity implied by either its depth
or its area. Thresholds are loaded from ``configs/default.yaml``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class SeverityBucket:
    name: str
    max_depth_m: float
    max_area_m2: float

    @classmethod
    def from_dict(cls, d: dict) -> "SeverityBucket":
        def _num(v):
            if v in (float("inf"), "inf", ".inf"):
                return math.inf
            return float(v)

        return cls(
            name=str(d["name"]),
            max_depth_m=_num(d["max_depth_m"]),
            max_area_m2=_num(d["max_area_m2"]),
        )


def _bucket_index(value: float, bounds: Sequence[float]) -> int:
    for i, upper in enumerate(bounds):
        if value <= upper:
            return i
    return len(bounds) - 1


def classify_pothole(
    depth_m: float,
    area_m2: float,
    buckets: Sequence[SeverityBucket],
    *,
    signal: str = "max",
) -> str:
    """Return the severity label.

    ``signal`` selects which bucketing rule is used:

    * ``"max"`` (default) - pick the *worse* of depth- and area-based bucketing.
      Matches GOST R 50597-2017's "any axis exceeds threshold" rule.
    * ``"area_only"`` - bucket by area, ignore depth entirely. Use when the
      depth signal is unreliable (e.g. monocular metric depth on
      out-of-distribution dashcam frames). Severity stays publishable.
    * ``"depth_only"`` - bucket by depth, ignore area.
    """
    if not buckets:
        raise ValueError("buckets must be non-empty")
    depth_bounds = [b.max_depth_m for b in buckets]
    area_bounds = [b.max_area_m2 for b in buckets]
    if signal == "max":
        idx = max(_bucket_index(depth_m, depth_bounds), _bucket_index(area_m2, area_bounds))
    elif signal == "area_only":
        idx = _bucket_index(area_m2, area_bounds)
    elif signal == "depth_only":
        idx = _bucket_index(depth_m, depth_bounds)
    else:
        raise ValueError(f"unknown signal {signal!r}")
    return buckets[idx].name


def make_severity_classifier(
    thresholds: Sequence[dict],
    *,
    signal: str = "max",
) -> Callable[[float, float], str]:
    """Factory returning a callable ``(depth_m, area_m2) -> str``.

    ``signal`` is forwarded to :func:`classify_pothole`.
    """
    buckets = [SeverityBucket.from_dict(d) for d in thresholds]
    return lambda depth_m, area_m2: classify_pothole(depth_m, area_m2, buckets, signal=signal)
