#!/usr/bin/env python3
"""
Paper-faithful precipitation transformations from Wang et al. (2026).

Reference:
Wang et al., Geosci. Model Dev., 19, 6027-6041, 2026.
https://doi.org/10.5194/gmd-19-6027-2026

The paper's comparative experiment uses:
  * displacement bias: horizontal integer-grid translations
  * intensity bias: multiplicative precipitation factor
  * area-size bias: geometric scaling around the centroid of precipitating grids

Important reproducibility note
------------------------------
The equations and interpolation rules below follow the published paper:
  - displacement preserves intensity and shifts grid coordinates
  - intensity multiplies p by a scalar
  - area size maps (i,j) about the centroid
  - enlargement uses bilinear interpolation to fill gaps
  - contraction averages values that map to the same output grid

The authors' released implementation should still be audited before calling any
implementation "bit-for-bit identical" to their code. This module is intended
to reproduce the published transformation definition transparently.
"""

from __future__ import annotations

import numpy as np


DISPLACEMENT_BIASES = np.arange(-10, 11, dtype=int)
INTENSITY_BIASES = np.array(
    [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,
     1.10,1.20,1.30,1.40,1.50,1.60,1.70,1.80,1.90,2.00],
    dtype=np.float32,
)
AREA_SIZE_BIASES = INTENSITY_BIASES.copy()


def displacement(field: np.ndarray, bias: int) -> np.ndarray:
    """
    Horizontal grid-cell shift. Positive bias shifts east/right.
    Vacated cells are zero-filled; values shifted outside the domain are lost.
    No periodic wrapping is used.
    """
    x = np.asarray(field, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("field must be 2-D")
    b = int(bias)
    out = np.zeros_like(x)
    if b == 0:
        return x.copy()
    if abs(b) >= x.shape[1]:
        return out
    if b > 0:
        out[:, b:] = x[:, :-b]
    else:
        k = -b
        out[:, :-k] = x[:, k:]
    return out


def intensity(field: np.ndarray, factor: float) -> np.ndarray:
    """Multiply every grid-cell precipitation amount by factor."""
    x = np.asarray(field, dtype=np.float32)
    return np.maximum(x * np.float32(factor), 0.0).astype(np.float32)


def _wet_centroid(field: np.ndarray, wet_epsilon: float = 0.0):
    mask = np.asarray(field) > float(wet_epsilon)
    if not np.any(mask):
        return None
    y, x = np.nonzero(mask)
    return float(y.mean()), float(x.mean())


def _bilinear_sample(field: np.ndarray, yy: np.ndarray, xx: np.ndarray):
    """Bilinear sampling with zero outside the original grid."""
    ny, nx = field.shape
    y0 = np.floor(yy).astype(np.int64)
    x0 = np.floor(xx).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = yy - y0
    wx = xx - x0

    def take(y, x):
        good = (y >= 0) & (y < ny) & (x >= 0) & (x < nx)
        vals = np.zeros_like(yy, dtype=np.float32)
        vals[good] = field[y[good], x[good]]
        return vals

    v00 = take(y0, x0)
    v01 = take(y0, x1)
    v10 = take(y1, x0)
    v11 = take(y1, x1)

    out = (
        (1-wy)*(1-wx)*v00
        + (1-wy)*wx*v01
        + wy*(1-wx)*v10
        + wy*wx*v11
    )
    return out.astype(np.float32)


def area_size(field: np.ndarray, scale: float, wet_epsilon: float = 0.0) -> np.ndarray:
    """
    Scale precipitation area about the centroid of precipitating grid cells.

    s >= 1:
      Use inverse-coordinate bilinear interpolation, matching the paper's
      instruction to fill gaps created by enlargement with bilinear values.

    s < 1:
      Forward-map precipitating grid cells to their scaled coordinates, round
      to the nearest output cell, and average intensities for collisions,
      matching the paper's contraction rule.

    Intensity values are not intentionally multiplied by area scaling.
    """
    x = np.asarray(field, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("field must be 2-D")
    s = float(scale)
    if s <= 0:
        raise ValueError("scale must be > 0")
    if np.isclose(s, 1.0):
        return x.copy()

    centroid = _wet_centroid(x, wet_epsilon=wet_epsilon)
    if centroid is None:
        return x.copy()
    cy, cx = centroid
    ny, nx = x.shape

    if s >= 1.0:
        oy, ox = np.indices((ny, nx), dtype=np.float64)
        # inverse of y' = s(y-cy)+cy
        iy = (oy - cy) / s + cy
        ix = (ox - cx) / s + cx
        out = _bilinear_sample(x, iy, ix)
        # Avoid creating tiny interpolation haze where the inverse location
        # lies entirely outside the original precip field.
        out[out < 0] = 0
        return out

    # Contraction: forward map only precipitating grids and average collisions.
    mask = x > float(wet_epsilon)
    ys, xs = np.nonzero(mask)
    vals = x[ys, xs].astype(np.float64)

    ynew = np.rint(s * (ys.astype(np.float64) - cy) + cy).astype(np.int64)
    xnew = np.rint(s * (xs.astype(np.float64) - cx) + cx).astype(np.int64)

    good = (ynew >= 0) & (ynew < ny) & (xnew >= 0) & (xnew < nx)
    ynew = ynew[good]
    xnew = xnew[good]
    vals = vals[good]

    sums = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(sums, (ynew, xnew), vals)
    np.add.at(counts, (ynew, xnew), 1)

    out = np.zeros((ny, nx), dtype=np.float32)
    hit = counts > 0
    out[hit] = (sums[hit] / counts[hit]).astype(np.float32)
    return out


def transform(field: np.ndarray, family: str, bias: float) -> np.ndarray:
    family = str(family).lower()
    if family == "displacement":
        return displacement(field, int(bias))
    if family == "intensity":
        return intensity(field, float(bias))
    if family in ("area", "area_size", "size"):
        return area_size(field, float(bias))
    raise ValueError(f"unknown transform family {family!r}")

