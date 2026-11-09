"""Metrics and helpers for the HRRR time-lag selection experiment."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter

def scalar_metrics(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {k: np.nan for k in [
            "rmse_mm","bcrmse_mm","mean_abs_diff_mm",
            "mean_bias_b_minus_a_mm","corr","mean_a_mm","mean_b_mm",
            "max_a_mm","max_b_mm"
        ]}
    aa = np.asarray(a[mask], dtype=np.float64)
    bb = np.asarray(b[mask], dtype=np.float64)
    diff = bb - aa
    bias = float(np.mean(diff))
    sa, sb = float(np.std(aa)), float(np.std(bb))
    corr = float(np.corrcoef(aa, bb)[0, 1]) if sa > 0 and sb > 0 else np.nan
    return {
        "rmse_mm": float(np.sqrt(np.mean(diff**2))),
        "bcrmse_mm": float(np.sqrt(np.mean((diff-bias)**2))),
        "mean_abs_diff_mm": float(np.mean(np.abs(diff))),
        "mean_bias_b_minus_a_mm": bias,
        "corr": corr,
        "mean_a_mm": float(np.mean(aa)),
        "mean_b_mm": float(np.mean(bb)),
        "max_a_mm": float(np.max(aa)),
        "max_b_mm": float(np.max(bb)),
    }

def fractions_skill_score(a, b, threshold_mm, window):
    if window < 1 or window % 2 != 1:
        raise ValueError("FSS window must be a positive odd integer")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return np.nan
    ea = ((a >= threshold_mm) & valid).astype(np.float32)
    eb = ((b >= threshold_mm) & valid).astype(np.float32)
    if window == 1:
        fa, fb, use = ea, eb, valid
    else:
        fa = uniform_filter(ea, size=window, mode="constant", cval=0.0)
        fb = uniform_filter(eb, size=window, mode="constant", cval=0.0)
        fv = uniform_filter(valid.astype(np.float32), size=window, mode="constant", cval=0.0)
        use = fv > 0.999999
    if not np.any(use):
        return np.nan
    faa, fbb = fa[use].astype(np.float64), fb[use].astype(np.float64)
    denom = np.mean(faa*faa + fbb*fbb)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.mean((faa-fbb)**2) / denom)

def fss_curve(a, b, threshold_mm, windows):
    return {int(w): fractions_skill_score(a, b, threshold_mm, int(w)) for w in windows}

def fss_column(window):
    return f"fss_w{int(window)}"

def wet_fraction(field, threshold_mm):
    finite = np.isfinite(field)
    return np.nan if not np.any(finite) else float(np.mean(field[finite] >= threshold_mm))

