#!/usr/bin/env python3
"""
Controlled precipitation morphology probes for representation diagnosis.

These are intentionally diagnostic transformations, not claims that real
forecast errors follow these exact mathematical operators.  Each family is
specified explicitly so that sensitivity can be attributed to a known spatial
change.

Families
--------
smoothing_sigma
    Gaussian smoothing. Total precipitation is renormalized to the original
    total after interpolation.

rotation_deg
    Rotate about the precipitation-weighted centroid. +/- rotations are used
    as variants and summarized together by magnitude. Total is renormalized.

elongation_ratio
    Area-preserving anisotropic scaling about the precipitation-weighted
    centroid: x*r, y/r or the orthogonal orientation. Two orientations are
    summarized together. Total is renormalized.

tile_fragmentation_cells
    Divide the field into 32x32-cell tiles and jitter every tile in a fixed,
    deterministic direction by d grid cells. This breaks spatial continuity
    while leaving the within-tile field unchanged. Total is renormalized.

object_dispersion_cells
    Identify >=0.1-mm 8-connected components and translate each sufficiently
    large component radially away from the precipitation centroid by d cells.
    Subthreshold/background pixels remain in place. If scipy labeling is not
    available, a deterministic quadrant-dispersion fallback is used. Total is
    renormalized.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, affine_transform, label
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


MORPHOLOGY_LEVELS = {
    "smoothing_sigma": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
    "rotation_deg": [0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0],
    "elongation_ratio": [1.0, 1.10, 1.25, 1.50, 1.75, 2.00],
    "tile_fragmentation_cells": [0, 1, 2, 4, 8, 12],
    "object_dispersion_cells": [0, 2, 4, 8, 12, 16],
}


def _x(field):
    a = np.asarray(field, dtype=np.float32)
    a = np.where(np.isfinite(a), a, 0.0)
    return np.maximum(a, 0.0)


def _wet_centroid(field, threshold=0.1):
    a = _x(field)
    w = np.where(a >= float(threshold), a, 0.0).astype(np.float64)
    s = w.sum()
    if s <= 0:
        return (0.5*(a.shape[0]-1), 0.5*(a.shape[1]-1))
    yy = np.arange(a.shape[0], dtype=np.float64)[:,None]
    xx = np.arange(a.shape[1], dtype=np.float64)[None,:]
    return float((w*yy).sum()/s), float((w*xx).sum()/s)


def _renorm_total(out, original):
    out = np.maximum(np.asarray(out, np.float32), 0.0)
    orig_sum = float(np.sum(original, dtype=np.float64))
    new_sum = float(np.sum(out, dtype=np.float64))
    if orig_sum > 0 and new_sum > 0:
        out = out * np.float32(orig_sum/new_sum)
    return np.maximum(out, 0.0).astype(np.float32)


def _require_scipy(name):
    if not HAVE_SCIPY:
        raise RuntimeError(f"{name} requires scipy.ndimage in this environment")


def smooth(field, sigma):
    a = _x(field)
    sigma = float(sigma)
    if sigma == 0:
        return a.copy()
    _require_scipy("smoothing")
    out = gaussian_filter(a, sigma=sigma, mode="constant", cval=0.0)
    return _renorm_total(out, a)


def _affine_about_center(field, forward_matrix_yx, center_yx):
    """Apply a forward y/x matrix about center using scipy inverse mapping."""
    _require_scipy("affine morphology transform")
    a = _x(field)
    Fm = np.asarray(forward_matrix_yx, dtype=np.float64)
    inv = np.linalg.inv(Fm)
    c = np.asarray(center_yx, dtype=np.float64)
    offset = c - inv @ c
    out = affine_transform(
        a,
        matrix=inv,
        offset=offset,
        output_shape=a.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return _renorm_total(out, a)


def rotate(field, degrees):
    a = _x(field)
    deg = float(degrees)
    if deg == 0:
        return a.copy()
    c = _wet_centroid(a)
    th = np.deg2rad(deg)
    # coordinates are [y,x]; this is a conventional 2-D rotation matrix
    mat = np.array([[np.cos(th), -np.sin(th)],
                    [np.sin(th),  np.cos(th)]], dtype=np.float64)
    return _affine_about_center(a, mat, c)


def elongate(field, ratio, orientation="x"):
    a = _x(field)
    r = float(ratio)
    if r <= 0:
        raise ValueError("ratio must be > 0")
    if np.isclose(r, 1.0):
        return a.copy()
    c = _wet_centroid(a)
    if orientation == "x":
        # y contracts by 1/r while x stretches by r; determinant = 1
        mat = np.array([[1.0/r, 0.0], [0.0, r]], dtype=np.float64)
    elif orientation == "y":
        mat = np.array([[r, 0.0], [0.0, 1.0/r]], dtype=np.float64)
    else:
        raise ValueError("orientation must be x or y")
    return _affine_about_center(a, mat, c)


def _shift_add(out, block, y0, x0):
    H, W = out.shape
    bh, bw = block.shape
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    dy0 = max(0, y0);  dx0 = max(0, x0)
    hh = min(bh-sy0, H-dy0)
    ww = min(bw-sx0, W-dx0)
    if hh > 0 and ww > 0:
        out[dy0:dy0+hh, dx0:dx0+ww] += block[sy0:sy0+hh, sx0:sx0+ww]


def tile_fragment(field, cells, seed, tile_size=32):
    a = _x(field)
    d = int(cells)
    if d == 0:
        return a.copy()
    H, W = a.shape
    out = np.zeros_like(a)
    rng = np.random.default_rng(int(seed))
    dirs = np.array([
        [-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]
    ], dtype=int)
    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            block = a[y0:min(H,y0+tile_size), x0:min(W,x0+tile_size)]
            dy, dx = dirs[int(rng.integers(0, len(dirs)))] * d
            _shift_add(out, block, y0+int(dy), x0+int(dx))
    return _renorm_total(out, a)


def _shift_masked_component(out, a, mask, dy, dx):
    ys, xs = np.nonzero(mask)
    ny = ys + int(dy)
    nx = xs + int(dx)
    good = (ny >= 0) & (ny < a.shape[0]) & (nx >= 0) & (nx < a.shape[1])
    np.add.at(out, (ny[good], nx[good]), a[ys[good], xs[good]])


def _quadrant_dispersion(field, cells):
    a = _x(field)
    d = int(cells)
    if d == 0:
        return a.copy()
    cy, cx = _wet_centroid(a)
    yy, xx = np.indices(a.shape)
    out = np.zeros_like(a)
    quadrants = [
        ((yy < cy) & (xx < cx), -d, -d),
        ((yy < cy) & (xx >= cx), -d, +d),
        ((yy >= cy) & (xx < cx), +d, -d),
        ((yy >= cy) & (xx >= cx), +d, +d),
    ]
    for mask, dy, dx in quadrants:
        _shift_masked_component(out, a, mask, dy, dx)
    return _renorm_total(out, a)


def object_dispersion(field, cells, seed=0, threshold=0.1, min_pixels=3):
    a = _x(field)
    d = int(cells)
    if d == 0:
        return a.copy()
    if not HAVE_SCIPY:
        return _quadrant_dispersion(a, d)

    wet = a >= float(threshold)
    structure = np.ones((3,3), dtype=np.int8)
    labels, nlab = label(wet, structure=structure)
    if nlab < 2:
        # Dispersion is undefined for a single coherent object; leave it unchanged.
        return a.copy()

    global_cy, global_cx = _wet_centroid(a, threshold=threshold)
    moved = np.zeros_like(a)
    moved_mask = np.zeros_like(wet)
    rng = np.random.default_rng(int(seed))

    for lab in range(1, nlab+1):
        mask = labels == lab
        npx = int(mask.sum())
        if npx < min_pixels:
            continue
        ys, xs = np.nonzero(mask)
        weights = a[mask].astype(np.float64)
        sw = weights.sum()
        if sw <= 0:
            continue
        cy = float(np.sum(ys*weights)/sw)
        cx = float(np.sum(xs*weights)/sw)
        vy, vx = cy-global_cy, cx-global_cx
        norm = math.hypot(vy, vx)
        if norm < 1e-6:
            ang = float(rng.uniform(0, 2*np.pi))
            vy, vx = math.sin(ang), math.cos(ang)
            norm = 1.0
        dy = int(round(d * vy/norm))
        dx = int(round(d * vx/norm))
        _shift_masked_component(moved, a, mask, dy, dx)
        moved_mask |= mask

    # Leave subthreshold and tiny-component pixels in place.
    moved += np.where(~moved_mask, a, 0.0).astype(np.float32)
    return _renorm_total(moved, a)


def variants(field, family, severity, case_seed=0):
    """Return [(variant_name, transformed_field), ...]."""
    sev = float(severity)
    if family == "smoothing_sigma":
        return [("gaussian", smooth(field, sev))]
    if family == "rotation_deg":
        if sev == 0:
            return [("identity", _x(field).copy())]
        return [("plus", rotate(field, +sev)), ("minus", rotate(field, -sev))]
    if family == "elongation_ratio":
        if np.isclose(sev, 1.0):
            return [("identity", _x(field).copy())]
        return [
            ("stretch_x", elongate(field, sev, "x")),
            ("stretch_y", elongate(field, sev, "y")),
        ]
    if family == "tile_fragmentation_cells":
        return [("tile_jitter", tile_fragment(field, int(round(sev)), case_seed))]
    if family == "object_dispersion_cells":
        return [("radial", object_dispersion(field, int(round(sev)), seed=case_seed))]
    raise ValueError(f"Unknown morphology family {family!r}")

