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
) -> str:
    """Return the severity label. Picks the *worse* of depth- and area-based bucketing."""
    if not buckets:
        raise ValueError("buckets must be non-empty")
    depth_bounds = [b.max_depth_m for b in buckets]
    area_bounds = [b.max_area_m2 for b in buckets]
    worst = max(_bucket_index(depth_m, depth_bounds), _bucket_index(area_m2, area_bounds))
    return buckets[worst].name


def make_severity_classifier(
    thresholds: Sequence[dict],
) -> Callable[[float, float], str]:
    """Factory that returns a callable ``(depth_m, area_m2) -> str`` over the
    supplied threshold list (e.g. ``config["severity"]["thresholds"]``)."""
    buckets = [SeverityBucket.from_dict(d) for d in thresholds]
    return lambda depth_m, area_m2: classify_pothole(depth_m, area_m2, buckets)
