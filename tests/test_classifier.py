"""Tests for the GOST-based severity classifier."""
from __future__ import annotations

import pytest
import yaml
from pathlib import Path

from src.classifier import (
    SeverityBucket,
    classify_pothole,
    make_severity_classifier,
)


@pytest.fixture
def thresholds() -> list[dict]:
    return [
        {"name": "minor", "max_depth_m": 0.02, "max_area_m2": 0.05},
        {"name": "moderate", "max_depth_m": 0.05, "max_area_m2": 0.25},
        {"name": "major", "max_depth_m": 0.10, "max_area_m2": 1.00},
        {"name": "critical", "max_depth_m": ".inf", "max_area_m2": ".inf"},
    ]


def test_classifier_matches_depth_bucket(thresholds) -> None:
    classify = make_severity_classifier(thresholds)
    assert classify(0.01, 0.00) == "minor"
    assert classify(0.03, 0.00) == "moderate"
    assert classify(0.07, 0.00) == "major"
    assert classify(0.25, 0.00) == "critical"


def test_classifier_matches_area_bucket(thresholds) -> None:
    classify = make_severity_classifier(thresholds)
    assert classify(0.00, 0.03) == "minor"
    assert classify(0.00, 0.20) == "moderate"
    assert classify(0.00, 0.60) == "major"
    assert classify(0.00, 1.50) == "critical"


def test_classifier_picks_worst_of_two(thresholds) -> None:
    classify = make_severity_classifier(thresholds)
    # Depth → moderate, area → major → overall major.
    assert classify(0.03, 0.80) == "major"
    # Depth → critical, area → minor → overall critical.
    assert classify(0.30, 0.01) == "critical"


def test_boundary_values_are_inclusive(thresholds) -> None:
    classify = make_severity_classifier(thresholds)
    assert classify(0.02, 0.00) == "minor"
    assert classify(0.05, 0.05) == "moderate"
    assert classify(0.10, 0.25) == "major"


def test_config_yaml_thresholds_load() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    classify = make_severity_classifier(cfg["severity"]["thresholds"])
    assert classify(0.25, 2.0) == "critical"
    assert classify(0.0, 0.0) == "minor"


def test_severity_bucket_parses_inf_string() -> None:
    b = SeverityBucket.from_dict({"name": "x", "max_depth_m": ".inf", "max_area_m2": "inf"})
    assert b.max_depth_m == float("inf")
    assert b.max_area_m2 == float("inf")


def test_classify_pothole_rejects_empty_buckets() -> None:
    with pytest.raises(ValueError):
        classify_pothole(0.0, 0.0, [])


def test_area_only_signal_ignores_depth(thresholds) -> None:
    classify = make_severity_classifier(thresholds, signal="area_only")
    # Depth would say critical; area says moderate.
    assert classify(0.99, 0.20) == "moderate"
    # Area-only critical regardless of depth.
    assert classify(0.0, 1.50) == "critical"


def test_depth_only_signal_ignores_area(thresholds) -> None:
    classify = make_severity_classifier(thresholds, signal="depth_only")
    assert classify(0.03, 99.0) == "moderate"
    assert classify(0.99, 0.0) == "critical"


def test_unknown_signal_rejected(thresholds) -> None:
    with pytest.raises(ValueError):
        make_severity_classifier(thresholds, signal="bogus")(0.0, 0.0)
