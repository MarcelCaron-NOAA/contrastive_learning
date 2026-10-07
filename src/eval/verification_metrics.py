#!/usr/bin/env python3
"""Small, dependency-light precipitation verification metrics."""

from __future__ import annotations

import numpy as np


def mae(f, o):
    f = np.asarray(f, float); o = np.asarray(o, float)
    return float(np.mean(np.abs(f-o)))


def rmse(f, o):
    f = np.asarray(f, float); o = np.asarray(o, float)
    return float(np.sqrt(np.mean((f-o)**2)))


def pearson_r(f, o):
    f = np.asarray(f, float).ravel()
    o = np.asarray(o, float).ravel()
    if np.std(f) == 0 or np.std(o) == 0:
        return np.nan
    return float(np.corrcoef(f, o)[0,1])


def contingency(f, o, threshold):
    fb = np.asarray(f) >= float(threshold)
    ob = np.asarray(o) >= float(threshold)
    hits = int(np.sum(fb & ob))
    misses = int(np.sum((~fb) & ob))
    false_alarms = int(np.sum(fb & (~ob)))
    correct_neg = int(np.sum((~fb) & (~ob)))
    return hits, misses, false_alarms, correct_neg


def categorical_scores(f, o, threshold):
    h, m, fa, cn = contingency(f, o, threshold)
    ts_d = h + m + fa
    pod_d = h + m
    far_d = h + fa
    return {
        "hits": h, "misses": m, "false_alarms": fa, "correct_negatives": cn,
        "ts": np.nan if ts_d == 0 else h / ts_d,
        "pod": np.nan if pod_d == 0 else h / pod_d,
        "far": np.nan if far_d == 0 else fa / far_d,
        "normalized_far": np.nan if far_d == 0 else 1.0 - fa / far_d,
    }


def _box_fraction(binary, width):
    """Fraction in centered square neighborhood; scipy if available."""
    b = np.asarray(binary, dtype=np.float64)
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(b, size=int(width), mode="constant", cval=0.0)
    except Exception:
        # Integral-image fallback.
        w = int(width)
        before = w // 2
        after = w - 1 - before
        p = np.pad(b, ((before, after), (before, after)), mode="constant")
        s = np.pad(p, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)
        total = s[w:,w:] - s[:-w,w:] - s[w:,:-w] + s[:-w,:-w]
        return total / float(w*w)


def fss(f, o, threshold, width):
    ff = _box_fraction(np.asarray(f) >= float(threshold), width)
    oo = _box_fraction(np.asarray(o) >= float(threshold), width)
    mse = np.mean((ff-oo)**2)
    ref = np.mean(ff**2 + oo**2)
    if ref == 0:
        return np.nan
    return float(1.0 - mse/ref)


def continuous_and_wang_metrics(f, o, threshold):
    d = {
        "mae": mae(f,o),
        "rmse": rmse(f,o),
        "corr": pearson_r(f,o),
    }
    d.update(categorical_scores(f,o,threshold))
    d["fss5"] = fss(f,o,threshold,5)
    d["fss10"] = fss(f,o,threshold,10)
    return d

