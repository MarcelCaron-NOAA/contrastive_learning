#!/usr/bin/env python3
"""
Signed controlled morphology perturbations for v2 external diagnosis.

The sign is intentionally explicit so the plots can be centered on an identity
state, analogous to the original Wang-style bias plots.

Families
--------
smoothing_sharpening
    bias > 0: Gaussian smoothing with sigma=bias.
    bias < 0: unsharp enhancement using sigma=abs(bias): 2*x - G_sigma(x),
              clipped nonnegative and total-precipitation renormalized.
    This is a signed diagnostic axis, not a mathematical inverse pair.

rotation_deg
    Signed rotation about the precipitation-weighted centroid.

principal_axis_log2_scale
    Area-preserving anisotropic deformation aligned with the precipitation-
    weighted principal axes.  The plotted bias is log2(scale along the original
    major axis).  Negative values compress the original major axis; positive
    values elongate it.  The orthogonal axis receives the reciprocal scale so
    determinant=1.  Examples: -1 -> 0.5x, +1 -> 2x.

tile_fragmentation_cells
    The absolute bias is the tile-jitter displacement in grid cells. Positive
    and negative values use the same deterministic per-tile direction vectors
    with opposite sign. Both sides therefore represent equal-magnitude
    fragmentation in opposite displacement directions; negative does NOT mean
    "less fragmented".

object_dispersion_cells
    bias > 0: connected objects move radially away from the precipitation
              centroid (dispersion).
    bias < 0: connected objects move radially toward the centroid
              (concentration).

All geometry-changing operators are renormalized to preserve domain-total
precipitation after clipping/interpolation. These are diagnostic mathematical
probes, not claims that real forecast errors follow these exact operators.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, affine_transform, label
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


SIGNED_LEVELS = {
    "smoothing_sharpening": [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0,
                              0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
    "rotation_deg": [-90.0, -60.0, -45.0, -30.0, -20.0, -10.0, -5.0, 0.0,
                      5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0],
    "principal_axis_log2_scale": [-1.0, -0.75, -0.5, -0.25, 0.0,
                                   0.25, 0.5, 0.75, 1.0],
    "tile_fragmentation_cells": [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12],
    "object_dispersion_cells": [-16, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16],
}


def _x(field):
    a = np.asarray(field, dtype=np.float32)
    a = np.where(np.isfinite(a), a, 0.0)
    return np.maximum(a, 0.0)


def _wet_centroid(field, threshold=0.1):
    a = _x(field)
    w = np.where(a >= float(threshold), a, 0.0).astype(np.float64)
    s = float(w.sum())
    if s <= 0:
        return (0.5 * (a.shape[0] - 1), 0.5 * (a.shape[1] - 1))
    yy = np.arange(a.shape[0], dtype=np.float64)[:, None]
    xx = np.arange(a.shape[1], dtype=np.float64)[None, :]
    return float((w * yy).sum() / s), float((w * xx).sum() / s)


def _renorm_total(out, original):
    out = np.maximum(np.asarray(out, np.float32), 0.0)
    osum = float(np.sum(original, dtype=np.float64))
    nsum = float(np.sum(out, dtype=np.float64))
    if osum > 0 and nsum > 0:
        out = out * np.float32(osum / nsum)
    return np.maximum(out, 0.0).astype(np.float32)


def _require_scipy(name):
    if not HAVE_SCIPY:
        raise RuntimeError(f"{name} requires scipy.ndimage")


def smooth_or_sharpen(field, bias):
    a = _x(field)
    b = float(bias)
    if np.isclose(b, 0.0):
        return a.copy()
    _require_scipy("smoothing/sharpening")
    blur = gaussian_filter(a, sigma=abs(b), mode="constant", cval=0.0)
    if b > 0:
        out = blur
    else:
        # Simple unsharp enhancement. This is a diagnostic counterpart to
        # smoothing, not an exact inverse of Gaussian convolution.
        out = 2.0 * a - blur
    return _renorm_total(out, a)


def _affine_about_center(field, forward_matrix_yx, center_yx):
    _require_scipy("affine morphology transform")
    a = _x(field)
    fwd = np.asarray(forward_matrix_yx, dtype=np.float64)
    inv = np.linalg.inv(fwd)
    c = np.asarray(center_yx, dtype=np.float64)
    offset = c - inv @ c
    out = affine_transform(
        a, matrix=inv, offset=offset, output_shape=a.shape,
        order=1, mode="constant", cval=0.0, prefilter=False,
    )
    return _renorm_total(out, a)


def rotate(field, degrees):
    a = _x(field)
    deg = float(degrees)
    if np.isclose(deg, 0.0):
        return a.copy()
    c = _wet_centroid(a)
    th = np.deg2rad(deg)
    mat = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]], dtype=np.float64)
    return _affine_about_center(a, mat, c)


def _principal_axes_yx(field, threshold=0.1):
    """Return center and orthonormal [major,minor] axes in y/x coordinates."""
    a = _x(field)
    cy, cx = _wet_centroid(a, threshold=threshold)
    w = np.where(a >= float(threshold), a, 0.0).astype(np.float64)
    if float(w.sum()) <= 0:
        return (cy, cx), np.eye(2, dtype=np.float64)
    yy, xx = np.indices(a.shape, dtype=np.float64)
    dy = yy - cy
    dx = xx - cx
    sw = float(w.sum())
    cov = np.array([
        [(w * dy * dy).sum() / sw, (w * dy * dx).sum() / sw],
        [(w * dy * dx).sum() / sw, (w * dx * dx).sum() / sw],
    ], dtype=np.float64)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    # Columns are orthonormal major/minor axes.
    return (cy, cx), vecs


def principal_axis_deform(field, log2_scale):
    a = _x(field)
    b = float(log2_scale)
    if np.isclose(b, 0.0):
        return a.copy()
    scale_major = float(2.0 ** b)
    scale_minor = 1.0 / scale_major
    center, axes = _principal_axes_yx(a)
    local = np.diag([scale_major, scale_minor])
    mat = axes @ local @ axes.T
    return _affine_about_center(a, mat, center)


def _shift_add(out, block, y0, x0):
    H, W = out.shape
    bh, bw = block.shape
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    dy0 = max(0, y0);  dx0 = max(0, x0)
    hh = min(bh - sy0, H - dy0)
    ww = min(bw - sx0, W - dx0)
    if hh > 0 and ww > 0:
        out[dy0:dy0+hh, dx0:dx0+ww] += block[sy0:sy0+hh, sx0:sx0+ww]


def signed_tile_fragment(field, signed_cells, seed, tile_size=32):
    a = _x(field)
    signed = int(round(float(signed_cells)))
    if signed == 0:
        return a.copy()
    d = abs(signed)
    sgn = 1 if signed > 0 else -1
    H, W = a.shape
    out = np.zeros_like(a)
    rng = np.random.default_rng(int(seed))
    dirs = np.array([
        [-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]
    ], dtype=int)
    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            block = a[y0:min(H, y0+tile_size), x0:min(W, x0+tile_size)]
            dy, dx = dirs[int(rng.integers(0, len(dirs)))] * d * sgn
            _shift_add(out, block, y0 + int(dy), x0 + int(dx))
    return _renorm_total(out, a)


def _shift_masked_component(out, a, mask, dy, dx):
    ys, xs = np.nonzero(mask)
    ny = ys + int(dy)
    nx = xs + int(dx)
    good = (ny >= 0) & (ny < a.shape[0]) & (nx >= 0) & (nx < a.shape[1])
    np.add.at(out, (ny[good], nx[good]), a[ys[good], xs[good]])


def signed_object_dispersion(field, signed_cells, seed=0, threshold=0.1, min_pixels=3):
    a = _x(field)
    signed = int(round(float(signed_cells)))
    if signed == 0:
        return a.copy()
    _require_scipy("signed object dispersion")
    d = abs(signed)
    direction = 1 if signed > 0 else -1

    wet = a >= float(threshold)
    labels, nlab = label(wet, structure=np.ones((3,3), dtype=np.int8))
    if nlab < 2:
        return a.copy()

    global_cy, global_cx = _wet_centroid(a, threshold=threshold)
    moved = np.zeros_like(a)
    moved_mask = np.zeros_like(wet)
    rng = np.random.default_rng(int(seed))

    for lab in range(1, nlab + 1):
        mask = labels == lab
        if int(mask.sum()) < min_pixels:
            continue
        ys, xs = np.nonzero(mask)
        weights = a[mask].astype(np.float64)
        sw = float(weights.sum())
        if sw <= 0:
            continue
        cy = float(np.sum(ys * weights) / sw)
        cx = float(np.sum(xs * weights) / sw)
        vy, vx = cy - global_cy, cx - global_cx
        norm = math.hypot(vy, vx)
        if norm < 1e-6:
            ang = float(rng.uniform(0, 2*np.pi))
            vy, vx = math.sin(ang), math.cos(ang)
            norm = 1.0
        dy = int(round(direction * d * vy / norm))
        dx = int(round(direction * d * vx / norm))
        _shift_masked_component(moved, a, mask, dy, dx)
        moved_mask |= mask

    moved += np.where(~moved_mask, a, 0.0).astype(np.float32)
    return _renorm_total(moved, a)


def transform_signed(field, family, bias, case_seed=0):
    if family == "smoothing_sharpening":
        return smooth_or_sharpen(field, bias)
    if family == "rotation_deg":
        return rotate(field, bias)
    if family == "principal_axis_log2_scale":
        return principal_axis_deform(field, bias)
    if family == "tile_fragmentation_cells":
        return signed_tile_fragment(field, bias, seed=case_seed)
    if family == "object_dispersion_cells":
        return signed_object_dispersion(field, bias, seed=case_seed)
    raise ValueError(f"Unknown signed morphology family {family!r}")

