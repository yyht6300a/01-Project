"""Numpy-only ranking / calibration metrics for the risk model."""
from __future__ import annotations

import numpy as np


def auroc(y_true, y_score) -> float | None:
    """Area under the ROC curve via the rank (Mann-Whitney U) formula.

    Returns None when undefined (empty input or a single class present).
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    if y.size == 0 or s.size == 0 or y.min() == y.max():
        return None
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(y_true, y_score) -> float | None:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    if y.size == 0 or y.min() == y.max():
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    recall = tp / tp[-1]
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def prf_at(y_true, y_score, threshold: float) -> dict:
    """Precision / recall / F1 for the positive (unsafe) class at a threshold."""
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    pred = (s >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def ece(y_true, y_prob, bins: int = 10) -> float | None:
    """Expected calibration error over equal-width probability bins."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i < bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def best_f1_threshold(y_true, y_score, lo: float = 0.05, hi: float = 0.95, step: float = 0.01):
    """Return (threshold, metrics) maximizing F1 on the positive class."""
    best_t, best = 0.5, None
    t = lo
    while t <= hi + 1e-9:
        m = prf_at(y_true, y_score, t)
        if best is None or m["f1"] > best["f1"]:
            best_t, best = t, m
        t = round(t + step, 6)
    return best_t, best or {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

