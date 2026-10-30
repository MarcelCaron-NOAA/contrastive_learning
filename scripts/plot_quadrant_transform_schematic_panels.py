#!/usr/bin/env python3
"""
Create two 3x3 schematic morphology-panel figures using a synthetic field
whose top-left quadrant is 1.0 and the rest of the domain is 0.0.

Outputs:
  - quadrant_negative_transformations_3x3.png
  - quadrant_positive_transformations_3x3.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, affine_transform, label


# ============================================================================
# Basic utilities
# ============================================================================

def _x(field: np.ndarray) -> np.ndarray:
    a = np.asarray(field, dtype=np.float32)
    a = np.where(np.isfinite(a), a, 0.0)
    return np.maximum(a, 0.0)


def _wet_centroid_binary(field: np.ndarray, wet_epsilon: float = 0.0):
    mask = np.asarray(field) > float(wet_epsilon)
    if not np.any(mask):
        return None
    y, x = np.nonzero(mask)
    return float(y.mean()), float(x.mean())


def _wet_centroid_weighted(field: np.ndarray, threshold: float = 0.1):
    a = _x(field)
    w = np.where(a >= float(threshold), a, 0.0).astype(np.float64)
    s = float(w.sum())
    if s <= 0:
        return (0.5 * (a.shape[0] - 1), 0.5 * (a.shape[1] - 1))
    yy = np.arange(a.shape[0], dtype=np.float64)[:, None]
    xx = np.arange(a.shape[1], dtype=np.float64)[None, :]
    return float((w * yy).sum() / s), float((w * xx).sum() / s)


def _renorm_total(out: np.ndarray, original: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(out, np.float32), 0.0)
    osum = float(np.sum(original, dtype=np.float64))
    nsum = float(np.sum(out, dtype=np.float64))
    if osum > 0 and nsum > 0:
        out = out * np.float32(osum / nsum)
    return np.maximum(out, 0.0).astype(np.float32)


def _bilinear_sample(field: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
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
        (1 - wy) * (1 - wx) * v00
        + (1 - wy) * wx * v01
        + wy * (1 - wx) * v10
        + wy * wx * v11
    )
    return out.astype(np.float32)


# ============================================================================
# Wang-style transforms
# ============================================================================

def displacement(field: np.ndarray, bias: int) -> np.ndarray:
    x = np.asarray(field, dtype=np.float32)
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
    x = np.asarray(field, dtype=np.float32)
    return np.maximum(x * np.float32(factor), 0.0).astype(np.float32)


def area_size(field: np.ndarray, scale: float, wet_epsilon: float = 0.0) -> np.ndarray:
    x = np.asarray(field, dtype=np.float32)
    s = float(scale)
    if np.isclose(s, 1.0):
        return x.copy()

    centroid = _wet_centroid_binary(x, wet_epsilon=wet_epsilon)
    if centroid is None:
        return x.copy()
    cy, cx = centroid
    ny, nx = x.shape

    # enlargement
    if s >= 1.0:
        oy, ox = np.indices((ny, nx), dtype=np.float64)
        iy = (oy - cy) / s + cy
        ix = (ox - cx) / s + cx
        out = _bilinear_sample(x, iy, ix)
        out[out < 0] = 0
        return out

    # shrinking
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


# ============================================================================
# Morphology-style transforms
# ============================================================================

def smooth_or_sharpen(field: np.ndarray, bias: float) -> np.ndarray:
    a = _x(field)
    b = float(bias)
    if np.isclose(b, 0.0):
        return a.copy()
    blur = gaussian_filter(a, sigma=abs(b), mode="constant", cval=0.0)
    out = blur if b > 0 else (2.0 * a - blur)
    return _renorm_total(out, a)


def _affine_about_center(field: np.ndarray, forward_matrix_yx: np.ndarray, center_yx):
    a = _x(field)
    fwd = np.asarray(forward_matrix_yx, dtype=np.float64)
    inv = np.linalg.inv(fwd)
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


def rotate(field: np.ndarray, degrees: float) -> np.ndarray:
    a = _x(field)
    deg = float(degrees)
    if np.isclose(deg, 0.0):
        return a.copy()
    c = _wet_centroid_weighted(a)
    th = np.deg2rad(deg)
    mat = np.array(
        [
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)],
        ],
        dtype=np.float64,
    )
    return _affine_about_center(a, mat, c)


def _principal_axes_yx(field: np.ndarray, threshold: float = 0.1):
    a = _x(field)
    cy, cx = _wet_centroid_weighted(a, threshold=threshold)
    w = np.where(a >= float(threshold), a, 0.0).astype(np.float64)
    if float(w.sum()) <= 0:
        return (cy, cx), np.eye(2, dtype=np.float64)

    yy, xx = np.indices(a.shape, dtype=np.float64)
    dy = yy - cy
    dx = xx - cx
    sw = float(w.sum())
    cov = np.array(
        [
            [(w * dy * dy).sum() / sw, (w * dy * dx).sum() / sw],
            [(w * dy * dx).sum() / sw, (w * dx * dx).sum() / sw],
        ],
        dtype=np.float64,
    )
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    return (cy, cx), vecs


def principal_axis_deform(field: np.ndarray, log2_scale: float) -> np.ndarray:
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


def _shift_add(out: np.ndarray, block: np.ndarray, y0: int, x0: int):
    H, W = out.shape
    bh, bw = block.shape
    sy0 = max(0, -y0)
    sx0 = max(0, -x0)
    dy0 = max(0, y0)
    dx0 = max(0, x0)
    hh = min(bh - sy0, H - dy0)
    ww = min(bw - sx0, W - dx0)
    if hh > 0 and ww > 0:
        out[dy0:dy0 + hh, dx0:dx0 + ww] += block[sy0:sy0 + hh, sx0:sx0 + ww]


def signed_tile_fragment(
    field: np.ndarray,
    signed_cells: int,
    seed: int,
    tile_size: int = 32,
) -> np.ndarray:
    a = _x(field)
    signed_cells = int(round(float(signed_cells)))
    if signed_cells == 0:
        return a.copy()

    d = abs(signed_cells)
    sgn = 1 if signed_cells > 0 else -1
    H, W = a.shape
    out = np.zeros_like(a)
    rng = np.random.default_rng(int(seed))
    dirs = np.array(
        [[-1, -1], [-1, 0], [-1, 1],
         [ 0, -1],          [ 0, 1],
         [ 1, -1], [ 1, 0], [ 1, 1]],
        dtype=int,
    )

    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            block = a[y0:min(H, y0 + tile_size), x0:min(W, x0 + tile_size)]
            dy, dx = dirs[int(rng.integers(0, len(dirs)))] * d * sgn
            _shift_add(out, block, y0 + int(dy), x0 + int(dx))

    return _renorm_total(out, a)


def _shift_masked_component(out: np.ndarray, a: np.ndarray, mask: np.ndarray, dy: int, dx: int):
    ys, xs = np.nonzero(mask)
    ny = ys + int(dy)
    nx = xs + int(dx)
    good = (ny >= 0) & (ny < a.shape[0]) & (nx >= 0) & (nx < a.shape[1])
    np.add.at(out, (ny[good], nx[good]), a[ys[good], xs[good]])


def signed_object_dispersion(
    field: np.ndarray,
    signed_cells: int,
    seed: int = 0,
    threshold: float = 0.1,
    min_pixels: int = 3,
) -> np.ndarray:
    a = _x(field)
    signed_cells = int(round(float(signed_cells)))
    if signed_cells == 0:
        return a.copy()

    d = abs(signed_cells)
    direction = 1 if signed_cells > 0 else -1

    wet = a >= float(threshold)
    labels, nlab = label(wet, structure=np.ones((3, 3), dtype=np.int8))
    if nlab < 2:
        return a.copy()

    global_cy, global_cx = _wet_centroid_weighted(a, threshold=threshold)
    moved = np.zeros_like(a)
    moved_mask = np.zeros_like(wet)
    rng = np.random.default_rng(int(seed))

    for lab_id in range(1, nlab + 1):
        mask = labels == lab_id
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
            ang = float(rng.uniform(0, 2 * np.pi))
            vy, vx = math.sin(ang), math.cos(ang)
            norm = 1.0

        dy = int(round(direction * d * vy / norm))
        dx = int(round(direction * d * vx / norm))

        _shift_masked_component(moved, a, mask, dy, dx)
        moved_mask |= mask

    moved += np.where(~moved_mask, a, 0.0).astype(np.float32)
    return _renorm_total(moved, a)


# ============================================================================
# Synthetic base field
# ============================================================================

def build_quadrant_field(ny: int = 256, nx: int = 256, value: float = 1.0) -> np.ndarray:
    """
    Build a field whose top-left quadrant (when plotted with origin='lower')
    has value 1.0 and all other pixels are 0.0.
    """
    base = np.zeros((ny, nx), dtype=np.float32)
    cy = ny // 2
    cx = nx // 2
    base[cy:, :cx] = np.float32(value)
    return base


# ============================================================================
# Plotting
# ============================================================================

def build_cmap():
    # white = 0, lighter reds < 1, darker reds > 1
    return LinearSegmentedColormap.from_list(
        "white_to_darkred",
        [
            (0.00, "#ffffff"),
            (0.35, "#f7b3b3"),
            (0.50, "#ff4d4d"),
            (0.75, "#cc0000"),
            (1.00, "#5c0000"),
        ],
    )


def make_panel_figure(panels, suptitle: str, outfile: Path, vmax: float):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()
    cmap = build_cmap()
    im = None

    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal", fraction=0.04, pad=0.03)
    cbar.set_label("Magnitude", fontsize=12, fontweight="bold")
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory in which to save output PNGs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic transforms.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = build_quadrant_field()
    seed = args.seed

    neg_panels = [
        ("Original", base),
        ("Displacement\nbias = -10 cells", displacement(base, -10)),
        ("Intensity\nfactor = 0.50", intensity(base, 0.50)),
        ("Area size\nscale = 0.50", area_size(base, 0.50)),
        ("Sharpening\nbias = -3.0", smooth_or_sharpen(base, -3.0)),
        ("Object concentration\nbias = -12 cells", signed_object_dispersion(base, -12, seed=seed)),
        ("Principal-axis compression\nlog2 scale = -1.0", principal_axis_deform(base, -1.0)),
        ("Rotation\nbias = -45°", rotate(base, -45.0)),
        ("Tile fragmentation\nbias = -8 cells", signed_tile_fragment(base, -8, seed=seed)),
    ]

    pos_panels = [
        ("Original", base),
        ("Displacement\nbias = +10 cells", displacement(base, +10)),
        ("Intensity\nfactor = 2.00", intensity(base, 2.00)),
        ("Area size\nscale = 2.00", area_size(base, 2.00)),
        ("Smoothing\nbias = +3.0", smooth_or_sharpen(base, +3.0)),
        ("Object dispersion\nbias = +12 cells", signed_object_dispersion(base, +12, seed=seed)),
        ("Principal-axis elongation\nlog2 scale = +1.0", principal_axis_deform(base, +1.0)),
        ("Rotation\nbias = +45°", rotate(base, +45.0)),
        ("Tile fragmentation\nbias = +8 cells", signed_tile_fragment(base, +8, seed=seed)),
    ]

    all_fields = [arr for _, arr in neg_panels + pos_panels]
    vmax = max(2.0, max(float(np.nanmax(a)) for a in all_fields))

    make_panel_figure(
        neg_panels,
        "Negative-direction representative transformations of a top-left-quadrant synthetic field",
        args.output_dir / "quadrant_negative_transformations_3x3.png",
        vmax=vmax,
    )
    make_panel_figure(
        pos_panels,
        "Positive-direction representative transformations of a top-left-quadrant synthetic field",
        args.output_dir / "quadrant_positive_transformations_3x3.png",
        vmax=vmax,
    )

    print("Wrote:")
    print(args.output_dir / "quadrant_negative_transformations_3x3.png")
    print(args.output_dir / "quadrant_positive_transformations_3x3.png")


if __name__ == "__main__":
    main()
