"""Evaluation metrics for the pothole measurement pipeline.

Implemented without scikit-learn so the metrics module stays lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mean_abs_rel_err: float        # mean |pred - gt| / |gt| (skips gt == 0)
    n: int


def regression_metrics(preds: Sequence[float], gts: Sequence[float]) -> RegressionMetrics:
    if len(preds) != len(gts):
        raise ValueError(f"length mismatch: {len(preds)} vs {len(gts)}")
    if not preds:
        return RegressionMetrics(0.0, 0.0, 0.0, 0)
    p = np.asarray(preds, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    diff = p - g
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mask = np.abs(g) > 1e-9
    rel = float(np.mean(np.abs(diff[mask]) / np.abs(g[mask]))) if mask.any() else 0.0
    return RegressionMetrics(mae=mae, rmse=rmse, mean_abs_rel_err=rel, n=len(p))


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: dict[str, dict[str, int]]
    n: int


def _confusion(y_true, y_pred, labels) -> dict[str, dict[str, int]]:
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        if t not in matrix:
            matrix[t] = {b: 0 for b in labels}
        if p not in matrix[t]:
            matrix[t][p] = 0
        matrix[t][p] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return ClassificationMetrics(0.0, 0.0, {}, {}, 0)

    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / len(y_true)

    per_class_f1: dict[str, float] = {}
    for lbl in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lbl and b == lbl)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lbl and b == lbl)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lbl and b != lbl)
        denom = 2 * tp + fp + fn
        per_class_f1[lbl] = (2 * tp) / denom if denom else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        confusion=_confusion(y_true, y_pred, labels),
        n=len(y_true),
    )
