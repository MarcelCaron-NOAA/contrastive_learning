#!/usr/bin/env python3
"""
Compute training-derived precipitation normalization statistics.

Methodological choice:
  * sample ONLY eligible training anchors (A), uniformly without replacement;
  * choose one uniformly random native-grid crop per sampled anchor;
  * include ALL pixels, including exactly zero precipitation;
  * compute population mean/std for both raw QPF6 and log1p(QPF6).

Why anchors only?
  A, P, and N are the same physical HRRR QPF variable. Using anchors gives a
  training-only estimate that does not depend on arbitrary choices such as
  K negatives per sample or the negative-sampling ablation. That keeps the
  normalization fixed when comparing climo45 versus calendar_any.

The output JSON is directly compatible with:
    PrecipTransform.from_stats_json("raw_zscore", ...)
    PrecipTransform.from_stats_json("log1p_zscore", ...)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import HRRRArchive


GRID_NY = 1059
GRID_NX = 1799


def detect_anchor_id_column(df: pd.DataFrame) -> str:
    for name in ("anchor_id", "sample_id", "field_id"):
        if name in df.columns:
            return name
    raise RuntimeError(
        "Could not identify anchor ID column. Expected one of "
        "'anchor_id', 'sample_id', or 'field_id'."
    )


class RunningMoments:
    """Numerically stable streaming population moments."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, arr):
        x = np.asarray(arr, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return

        n = int(x.size)
        mean_b = float(x.mean())
        m2_b = float(np.sum((x - mean_b) ** 2, dtype=np.float64))

        if self.count == 0:
            self.count = n
            self.mean = mean_b
            self.m2 = m2_b
        else:
            delta = mean_b - self.mean
            new_count = self.count + n
            self.mean += delta * n / new_count
            self.m2 += m2_b + delta * delta * self.count * n / new_count
            self.count = new_count

        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))

    def as_dict(self):
        if self.count == 0:
            raise RuntimeError("No finite values accumulated.")
        variance = self.m2 / self.count
        return {
            "mean": float(self.mean),
            "std": float(np.sqrt(max(variance, 0.0))),
            "variance": float(max(variance, 0.0)),
            "count": int(self.count),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "ddof": 0,
        }


def default_paths():
    data_dir = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data_dir / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "manifest": (
            data_dir
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "output": (
            data_dir
            / "metadata/dataset_v1/precip_normalization_stats_v1.json"
        ),
    }


def main():
    defaults = default_paths()

    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(defaults["anchors"]))
    p.add_argument("--manifest", default=str(defaults["manifest"]))
    p.add_argument("--output", default=str(defaults["output"]))
    p.add_argument("--split", default="train")
    p.add_argument("--num-patches", type=int, default=5000)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--progress-every", type=int, default=250)
    args = p.parse_args()

    if args.num_patches < 1:
        raise SystemExit("--num-patches must be >= 1")
    if not (1 <= args.patch_size <= min(GRID_NY, GRID_NX)):
        raise SystemExit("Invalid --patch-size")

    anchors_path = Path(args.anchors)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    anchors = pd.read_csv(anchors_path, low_memory=False)
    id_col = detect_anchor_id_column(anchors)

    if "split" not in anchors.columns:
        raise RuntimeError("Eligible-anchor table has no split column.")

    anchors = anchors.loc[
        anchors["split"].astype(str) == str(args.split)
    ].copy()
    anchors[id_col] = anchors[id_col].astype(str)

    if anchors.empty:
        raise RuntimeError(f"No anchors found for split={args.split!r}")

    n_take = min(args.num_patches, len(anchors))
    if n_take < args.num_patches:
        print(
            f"WARNING: requested {args.num_patches:,} patches but only "
            f"{len(anchors):,} anchors exist; using all anchors."
        )

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(anchors), size=n_take, replace=False)

    raw_stats = RunningMoments()
    log_stats = RunningMoments()

    zero_pixels = 0
    wet_01 = 0
    wet_1 = 0
    wet_5 = 0
    dry_patches = 0
    finite_pixels = 0

    archive = HRRRArchive(manifest_path)

    print("=" * 76)
    print("TRAINING PRECIPITATION NORMALIZATION STATISTICS")
    print("=" * 76)
    print(f"Eligible anchors : {len(anchors):,}")
    print(f"Sampled anchors  : {n_take:,}")
    print(f"Patch size       : {args.patch_size} x {args.patch_size}")
    print(f"Split            : {args.split}")
    print(f"Seed             : {args.seed}")
    print(f"Output           : {output_path}")
    print()
    print("Sampling: uniform anchors without replacement + uniform crop origins")
    print("Pixels: all finite pixels, including zeros")
    print()

    t0 = time.perf_counter()

    try:
        for j, idx in enumerate(selected, start=1):
            row = anchors.iloc[int(idx)]
            sample_id = str(row[id_col])

            y0 = int(rng.integers(0, GRID_NY - args.patch_size + 1))
            x0 = int(rng.integers(0, GRID_NX - args.patch_size + 1))

            qpf = archive.read_patch(
                sample_id,
                y0,
                x0,
                args.patch_size,
            )

            finite = np.isfinite(qpf)
            if not finite.all():
                qpf = np.where(finite, qpf, 0.0).astype(np.float32)

            min_qpf = float(np.min(qpf))
            if min_qpf < -1.0e-5:
                raise RuntimeError(
                    f"Materially negative QPF {min_qpf:.6g} mm "
                    f"in sample {sample_id}"
                )

            qpf = np.maximum(qpf, 0.0).astype(np.float32, copy=False)
            logq = np.log1p(qpf).astype(np.float32, copy=False)

            raw_stats.update(qpf)
            log_stats.update(logq)

            finite_pixels += int(qpf.size)
            zero_pixels += int(np.count_nonzero(qpf == 0.0))
            wet_01 += int(np.count_nonzero(qpf >= 0.1))
            wet_1 += int(np.count_nonzero(qpf >= 1.0))
            wet_5 += int(np.count_nonzero(qpf >= 5.0))
            dry_patches += int(float(np.max(qpf)) == 0.0)

            if (
                j == 1
                or j % args.progress_every == 0
                or j == n_take
            ):
                elapsed = time.perf_counter() - t0
                rate = j / elapsed if elapsed > 0 else np.nan
                eta = (n_take - j) / rate if rate > 0 else np.nan
                print(
                    f"{j:5d}/{n_take:,} patches "
                    f"({100*j/n_take:5.1f}%) | "
                    f"{rate:5.2f} patches/s | "
                    f"ETA {eta/60:5.1f} min"
                )

    finally:
        archive.close()

    elapsed = time.perf_counter() - t0

    result = {
        "version": 1,
        "source_variable": "HRRR native-grid 6-h accumulated precipitation",
        "units": "mm",
        "split": str(args.split),
        "field_role_used_for_stats": "anchor",
        "sampling": {
            "eligible_anchor_count": int(len(anchors)),
            "sampled_anchor_count": int(n_take),
            "anchor_sampling": "uniform_without_replacement",
            "crop_sampling": "uniform_crop_origin",
            "patch_size": int(args.patch_size),
            "grid_shape": [GRID_NY, GRID_NX],
            "seed": int(args.seed),
        },
        "pixel_policy": {
            "zeros_included": True,
            "nonfinite_replaced_with_zero": True,
            "negative_values_clipped_to_zero_if_above_tolerance": True,
            "negative_error_tolerance_mm": -1.0e-5,
        },
        "raw": raw_stats.as_dict(),
        "log1p": log_stats.as_dict(),
        "diagnostics": {
            "pixel_count": int(finite_pixels),
            "zero_pixel_fraction": float(zero_pixels / finite_pixels),
            "wet_fraction_ge_0p1_mm": float(wet_01 / finite_pixels),
            "wet_fraction_ge_1_mm": float(wet_1 / finite_pixels),
            "wet_fraction_ge_5_mm": float(wet_5 / finite_pixels),
            "dry_patch_fraction": float(dry_patches / n_take),
            "dry_patch_count": int(dry_patches),
        },
        "runtime": {
            "seconds": float(elapsed),
            "patches_per_second": float(n_take / elapsed),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print()
    print("=" * 76)
    print("NORMALIZATION STATISTICS COMPLETE")
    print("=" * 76)
    print(
        f"Raw    : mean={result['raw']['mean']:.8f}, "
        f"std={result['raw']['std']:.8f}"
    )
    print(
        f"log1p  : mean={result['log1p']['mean']:.8f}, "
        f"std={result['log1p']['std']:.8f}"
    )
    print(
        f"Zero pixels      : "
        f"{100*result['diagnostics']['zero_pixel_fraction']:.2f}%"
    )
    print(
        f"Dry patches      : "
        f"{100*result['diagnostics']['dry_patch_fraction']:.2f}%"
    )
    print(f"Runtime          : {elapsed/60:.1f} min")
    print(f"Wrote            : {output_path}")


if __name__ == "__main__":
    main()

