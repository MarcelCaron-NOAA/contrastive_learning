#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


def find_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find any of {names}; columns={list(df.columns)}")


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "t", "yes"])


class ZarrCache:
    def __init__(self, max_open=24):
        self.max_open = max_open
        self.cache = OrderedDict()

    def open(self, path):
        path = str(path)
        if path in self.cache:
            store, root = self.cache.pop(path)
            self.cache[path] = (store, root)
            return root
        store = zarr.ZipStore(path, mode="r")
        root = zarr.open_consolidated(store, mode="r")
        self.cache[path] = (store, root)
        while len(self.cache) > self.max_open:
            _, (s, _) = self.cache.popitem(last=False)
            s.close()
        return root

    def close(self):
        for store, _ in self.cache.values():
            store.close()
        self.cache.clear()


def load_manifest(path):
    df = pd.read_csv(path, low_memory=False)
    sample_col = find_col(df, ["sample_id", "field_id", "id"])
    store_col = find_col(df, ["store", "store_path", "zarr_path", "path"])
    init_col = find_col(df, ["init_hour", "cycle_hour", "cycle"])
    fhr_col = find_col(df, ["forecast_hour", "lead_hour", "fhr", "lead"])
    if "available" in df.columns:
        df = df.loc[as_bool(df["available"])].copy()

    out = pd.DataFrame({
        "field_id": df[sample_col].astype(str),
        "store_path": df[store_col].astype(str),
        "init_hour": pd.to_numeric(df[init_col], errors="coerce"),
        "forecast_hour": pd.to_numeric(df[fhr_col], errors="coerce"),
    }).dropna()
    out["init_hour"] = out["init_hour"].astype(int)
    out["forecast_hour"] = out["forecast_hour"].astype(int)
    return out.set_index("field_id", drop=False)


def read_patch(row, cache, y0, x0, size):
    root = cache.open(row["store_path"])
    return np.asarray(
        root["qpf6"][
            int(row["init_hour"]),
            int(row["forecast_hour"]) - 6,
            y0:y0 + size,
            x0:x0 + size,
        ],
        dtype=np.float32,
    )


class StreamingMoments:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        self.n += len(x)
        self.sum += float(np.sum(x))
        self.sumsq += float(np.sum(x * x))

    def stats(self):
        mean = self.sum / self.n
        var = max(0.0, self.sumsq / self.n - mean * mean)
        return mean, np.sqrt(var)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--anchor-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-patches", type=int, default=5000)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--pixels-per-patch-for-quantiles", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260930)
    args = p.parse_args()

    manifest = load_manifest(Path(args.manifest))
    anchors = pd.read_csv(args.anchor_index, low_memory=False)
    train = anchors.loc[anchors["split"] == "train"].copy()
    if len(train) < args.n_patches:
        raise RuntimeError(f"Requested {args.n_patches} patches but only {len(train)} training anchors exist")

    rng = np.random.default_rng(args.seed)
    chosen = train.iloc[rng.choice(len(train), size=args.n_patches, replace=False)]

    raw_mom = StreamingMoments()
    log_mom = StreamingMoments()
    wet_samples = []
    zero_count = 0
    finite_count = 0
    cache = ZarrCache()

    try:
        for i, row in enumerate(chosen.itertuples(index=False), start=1):
            mrow = manifest.loc[row.anchor_id]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            y0 = int(rng.integers(0, 1059 - args.patch_size + 1))
            x0 = int(rng.integers(0, 1799 - args.patch_size + 1))
            patch = read_patch(mrow, cache, y0, x0, args.patch_size)

            vals = patch[np.isfinite(patch)]
            finite_count += len(vals)
            zero_count += int(np.sum(vals <= 0.0))
            raw_mom.update(vals)
            log_mom.update(np.log1p(np.maximum(vals, 0.0)))

            wet = vals[vals > 0.0]
            if len(wet):
                k = min(args.pixels_per_patch_for_quantiles, len(wet))
                idx = rng.choice(len(wet), size=k, replace=False)
                wet_samples.append(wet[idx].astype(np.float32))

            if i % 500 == 0:
                print(f"{i:,}/{args.n_patches:,} patches")
    finally:
        cache.close()

    wet = np.concatenate(wet_samples) if wet_samples else np.array([], dtype=np.float32)
    if len(wet) == 0:
        raise RuntimeError("No positive precipitation values sampled")

    raw_mean, raw_std = raw_mom.stats()
    log_mean, log_std = log_mom.stats()

    probs = np.array([0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 1.0])
    wet_q = np.quantile(wet, probs)

    stats = {
        "sampling": {
            "split": "train",
            "n_patches": args.n_patches,
            "patch_size": args.patch_size,
            "seed": args.seed,
            "finite_pixels": int(finite_count),
            "zero_or_less_pixels": int(zero_count),
            "dry_fraction": float(zero_count / finite_count),
            "wet_pixel_reservoir_n": int(len(wet)),
        },
        "raw": {"mean_all_pixels_mm": float(raw_mean), "std_all_pixels_mm": float(raw_std)},
        "log1p": {"mean_all_pixels": float(log_mean), "std_all_pixels": float(log_std),
                  "definition": "ln(1 + precipitation_mm)"},
        "wet_precipitation_quantiles_mm": {str(float(p_)): float(q_) for p_, q_ in zip(probs, wet_q)},
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preprocessing_stats_v1.json").write_text(json.dumps(stats, indent=2) + "\n")
    np.save(out_dir / "sampled_wet_values_v1.npy", wet)

    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    wet_sorted = np.sort(wet.astype(np.float64))
    ecdf = np.arange(1, len(wet_sorted) + 1) / len(wet_sorted)
    axes[0].plot(wet_sorted, ecdf, linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Positive 6-h precipitation (mm)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_title("(a)")
    axes[0].grid(alpha=0.2)

    xmax = max(100.0, float(np.quantile(wet, 0.999)))
    x = np.linspace(0.0, xmax, 800)
    z_raw = (x - raw_mean) / raw_std
    z_log = (np.log1p(x) - log_mean) / log_std
    axes[1].plot(x, z_raw, linewidth=2, label="Raw → z-score")
    axes[1].plot(x, z_log, linewidth=2, label="log1p → z-score")
    axes[1].set_xlabel("6-h precipitation (mm)")
    axes[1].set_ylabel("Standardized input value")
    axes[1].set_title("(b)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    raw_z_wet = (wet.astype(np.float64) - raw_mean) / raw_std
    log_z_wet = (np.log1p(wet.astype(np.float64)) - log_mean) / log_std
    order = np.argsort(wet)[::-1]
    raw_sq = raw_z_wet[order] ** 2
    log_sq = log_z_wet[order] ** 2
    top_fracs = np.array([0.001, 0.005, 0.01, 0.05, 0.10])
    raw_share, log_share = [], []
    for frac in top_fracs:
        n_top = max(1, int(np.ceil(frac * len(wet))))
        raw_share.append(np.sum(raw_sq[:n_top]) / np.sum(raw_sq))
        log_share.append(np.sum(log_sq[:n_top]) / np.sum(log_sq))
    axes[2].plot(top_fracs * 100, np.array(raw_share) * 100, marker="o", linewidth=2, label="Raw → z-score")
    axes[2].plot(top_fracs * 100, np.array(log_share) * 100, marker="o", linewidth=2, label="log1p → z-score")
    axes[2].set_xlabel("Highest-intensity wet pixels (%)")
    axes[2].set_ylabel("Share of wet-pixel squared input magnitude (%)")
    axes[2].set_title("(c)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False)

    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    fig.suptitle("Training-set precipitation transform diagnostic", fontsize=17, fontweight="bold")
    png = out_dir / "precipitation_transform_diagnostic_v1.png"
    pdf = out_dir / "precipitation_transform_diagnostic_v1.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("PRECIPITATION TRANSFORM DIAGNOSTIC COMPLETE")
    print("=" * 72)
    print(f"Patches                   : {args.n_patches:,}")
    print(f"Dry pixel fraction        : {zero_count / finite_count:.4f}")
    print(f"Raw mean/std (all pixels) : {raw_mean:.6f} / {raw_std:.6f} mm")
    print(f"log1p mean/std             : {log_mean:.6f} / {log_std:.6f}")
    print("Wet precipitation quantiles (mm):")
    for p_, q_ in zip(probs, wet_q):
        print(f"  {100*p_:7.3f}% : {q_:10.4f}")
    print(f"Wrote {out_dir / 'preprocessing_stats_v1.json'}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print("=" * 72)


if __name__ == "__main__":
    main()

