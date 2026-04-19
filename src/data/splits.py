"""Stratified train/val/test split."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence, TypeVar

T = TypeVar("T")


def _bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    return "5+"


def stratified_split(
    items: Sequence[T],
    key,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[T], list[T], list[T]]:
    """Split items into (train, val, test) stratified by `key(item)` bucket.

    For pothole samples, pass `key=lambda s: _bucket(len(s.boxes))` at the call site
    (kept generic here for reuse).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = random.Random(seed)
    groups: dict[object, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    train: list[T] = []
    val: list[T] = []
    test: list[T] = []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    return train, val, test


def box_count_bucket(sample) -> str:
    return _bucket(len(sample.boxes))
