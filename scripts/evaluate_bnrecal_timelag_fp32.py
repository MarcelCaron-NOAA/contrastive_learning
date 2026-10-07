#!/usr/bin/env python3
"""
Evaluate whether learned similarity degrades smoothly from 1 h -> 3 h -> 6 h.

Diagnostic design
-----------------
- held-out validation split only
- anchor lead fixed at F12
- same valid time for every comparison
- same geographic crop for all lagged fields belonging to one anchor
- both initialization directions are evaluated:
    earlier init: F13/F15/F18 for lag 1/3/6
    later init:   F11/F09/F06 for lag 1/3/6
- no negative fields are involved
- the model was trained only to regard +/-1 h as positive; +/-3 and +/-6 h
  provide an out-of-objective graded-similarity diagnostic

Primary diagnostic unit:
For each anchor, average the earlier/later direction similarities at each lag,
then ask whether mean similarity obeys 1 h > 3 h > 6 h.

Outputs:
  timelag_similarity_long_stepXXXX.csv
  timelag_similarity_anchor_stepXXXX.csv
  timelag_similarity_summary_stepXXXX.csv
  timelag_similarity_z_boxplot_stepXXXX.png
  timelag_similarity_h_boxplot_stepXXXX.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import ContrastiveIndex, HRRRArchive, PrecipTransform
from src.models.precip_resnet import PrecipResNet18


GRID_SHAPE = (1059, 1799)
LAGS = (1, 3, 6)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/full_epoch_train_validation_v1"
    return {
        "experiment_dir": exp,
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def stable_u32(text):
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=4
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def season_from_month(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def wet_p90(raw, threshold=0.1):
    x = np.asarray(raw, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    wet = x[x >= threshold]
    if len(wet) == 0:
        return 0.0
    return float(np.quantile(wet, 0.90))


class LagDataset(Dataset):
    def __init__(self, specs, manifest, transform, wet_threshold):
        self.specs = specs
        self.archive = HRRRArchive(manifest)
        self.transform = transform
        self.wet_threshold = float(wet_threshold)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        s = self.specs[i]
        a_raw = self.archive.read_patch(
            s["anchor_id"], s["y0"], s["x0"], s["patch_size"]
        )
        tensors = [self.transform(a_raw)]

        for lag in LAGS:
            for direction in ("earlier_init", "later_init"):
                sid = s[f"lag{lag}_{direction}_id"]
                raw = self.archive.read_patch(
                    sid, s["y0"], s["x0"], s["patch_size"]
                )
                tensors.append(self.transform(raw))

        meta = dict(s)
        meta["anchor_wet_p90_mm"] = wet_p90(
            a_raw, self.wet_threshold
        )
        return {
            "fields": torch.stack(tensors, dim=0),  # [7,1,H,W]
            "metadata": meta,
        }


def collate(items):
    return {
        "fields": torch.stack([x["fields"] for x in items], dim=0),
        "metadata": [x["metadata"] for x in items],
    }


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    spec = ckpt.get("model", {})
    model = PrecipResNet18(
        projection_hidden_dim=int(
            spec.get("projection_hidden_dim", 512)
        ),
        projection_dim=int(spec.get("projection_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def boxplot(df, col, path, title):
    data = [
        df.loc[df["lag_hours"] == lag, col].dropna().to_numpy()
        for lag in LAGS
    ]
    plt.figure(figsize=(8, 5.5))
    plt.boxplot(
        data,
        showfliers=False,
        whis=(5, 95),
    )
    # Set tick labels separately for compatibility across Matplotlib versions.
    plt.xticks(
        np.arange(1, len(LAGS) + 1),
        [f"{lag} h" for lag in LAGS],
    )
    plt.xlabel("Initialization time-lag magnitude")
    plt.ylabel("A / same-valid-time field cosine similarity")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(d["experiment_dir"]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--split", default="validation")
    p.add_argument("--anchor-fhr", type=int, default=12)
    p.add_argument("--sample-count", type=int, default=1000)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    args = p.parse_args()

    if args.anchor_fhr != 12:
        raise SystemExit(
            "This diagnostic is intentionally F12 in v1 so +/-6 h remains "
            "inside the archived F06-F18 lead range."
        )

    exp = Path(args.experiment_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else exp / "checkpoint_best.pt"
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this diagnostic.")

    device = torch.device("cuda")
    model, ckpt = load_model(ckpt_path, device)
    step = int(ckpt.get("step", -1))
    checkpoint_tag = ckpt_path.stem.replace("checkpoint_", "")

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )

    index = ContrastiveIndex(
        args.anchors,
        args.negative_pool,
        args.manifest,
    )
    anchors = index.anchors_for_split(args.split)
    anchors = anchors.loc[
        anchors["forecast_hour"] == args.anchor_fhr
    ].copy()

    lookup = {
        (int(r.valid_unix_s), int(r.forecast_hour)): str(r.sample_id)
        for r in index.manifest.itertuples()
    }

    eligible = []
    ny, nx = GRID_SHAPE

    for row in anchors.itertuples(index=False):
        valid_s = int(row.valid_unix_s)
        ids = {}
        ok = True

        for lag in LAGS:
            earlier = lookup.get((valid_s, args.anchor_fhr + lag))
            later = lookup.get((valid_s, args.anchor_fhr - lag))
            if earlier is None or later is None:
                ok = False
                break
            ids[f"lag{lag}_earlier_init_id"] = earlier
            ids[f"lag{lag}_later_init_id"] = later

        if not ok:
            continue

        crng = np.random.default_rng(
            np.random.SeedSequence([
                args.seed,
                stable_u32(str(row.anchor_id)),
            ])
        )
        y0 = int(crng.integers(0, ny - args.patch_size + 1))
        x0 = int(crng.integers(0, nx - args.patch_size + 1))

        ts = pd.Timestamp(row.valid_time)
        eligible.append({
            "anchor_id": str(row.anchor_id),
            "valid_time": str(row.valid_time),
            "valid_unix_s": int(row.valid_unix_s),
            "valid_hour_utc": int(row.valid_hour_utc),
            "season": season_from_month(ts.month),
            "anchor_fhr": int(row.forecast_hour),
            "y0": y0,
            "x0": x0,
            "patch_size": int(args.patch_size),
            **ids,
        })

    if len(eligible) < args.sample_count:
        print(
            f"Requested {args.sample_count} anchors but only "
            f"{len(eligible)} have all +/-1/3/6 fields; using all."
        )
        chosen = eligible
    else:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(
            len(eligible),
            size=args.sample_count,
            replace=False,
        )
        chosen = [eligible[int(i)] for i in idxs]

    dataset = LagDataset(
        chosen,
        args.manifest,
        transform,
        args.wet_threshold_mm,
    )

    kw = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    if args.num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**kw)

    long_rows = []

    print("=" * 82)
    print("FP32 TIME-LAG GRADATION DIAGNOSTIC")
    print("=" * 82)
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Checkpoint step : {step}")
    print(f"Split           : {args.split}")
    print(f"Anchor lead     : F{args.anchor_fhr:02d}")
    print(f"Anchors         : {len(chosen)}")
    print("Inference       : FP32 (autocast disabled)")
    print()

    field_order = []
    for lag in LAGS:
        for direction in ("earlier_init", "later_init"):
            field_order.append((lag, direction))

    with torch.no_grad():
        for ib, batch in enumerate(loader, start=1):
            fields = batch["fields"].to(
                device, non_blocking=True
            )  # [B,7,1,H,W]

            B = fields.shape[0]
            flat = fields.reshape(
                B * fields.shape[1], *fields.shape[2:]
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=False,
            ):
                h, z = model(flat)

            h = torch.nn.functional.normalize(h, dim=-1)
            h = h.reshape(B, 7, -1)
            z = z.reshape(B, 7, -1)

            h_a = h[:, 0]
            z_a = z[:, 0]

            h_sim = torch.einsum(
                "bd,bjd->bj", h_a, h[:, 1:]
            ).float().cpu().numpy()
            z_sim = torch.einsum(
                "bd,bjd->bj", z_a, z[:, 1:]
            ).float().cpu().numpy()

            for i, meta in enumerate(batch["metadata"]):
                for j, (lag, direction) in enumerate(field_order):
                    pair_id = meta[
                        f"lag{lag}_{direction}_id"
                    ]
                    long_rows.append({
                        "checkpoint_step": step,
                        "anchor_id": meta["anchor_id"],
                        "pair_id": pair_id,
                        "valid_time": meta["valid_time"],
                        "valid_hour_utc": meta["valid_hour_utc"],
                        "season": meta["season"],
                        "anchor_fhr": meta["anchor_fhr"],
                        "lag_hours": lag,
                        "direction": direction,
                        "z_similarity": float(z_sim[i, j]),
                        "h_similarity": float(h_sim[i, j]),
                        "anchor_wet_p90_mm": meta[
                            "anchor_wet_p90_mm"
                        ],
                        "crop_y0": meta["y0"],
                        "crop_x0": meta["x0"],
                    })

            if ib == 1 or ib % 25 == 0 or ib == len(loader):
                print(
                    f"Evaluated batch {ib:3d}/{len(loader)} "
                    f"({min(ib*args.batch_size, len(chosen)):4d} anchors)"
                )

    long_df = pd.DataFrame(long_rows)

    # Per-anchor mean over the two initialization directions.
    avg = (
        long_df.groupby(
            [
                "checkpoint_step",
                "anchor_id",
                "valid_time",
                "valid_hour_utc",
                "season",
                "anchor_fhr",
                "lag_hours",
            ],
            as_index=False,
        )[["z_similarity", "h_similarity"]]
        .mean()
        .rename(columns={
            "z_similarity": "z_similarity_direction_mean",
            "h_similarity": "h_similarity_direction_mean",
        })
    )

    pivot_z = avg.pivot(
        index="anchor_id",
        columns="lag_hours",
        values="z_similarity_direction_mean",
    )
    pivot_h = avg.pivot(
        index="anchor_id",
        columns="lag_hours",
        values="h_similarity_direction_mean",
    )

    meta_one = (
        long_df.sort_values("anchor_id")
        .drop_duplicates("anchor_id")[
            [
                "anchor_id",
                "valid_time",
                "valid_hour_utc",
                "season",
                "anchor_fhr",
                "anchor_wet_p90_mm",
                "crop_y0",
                "crop_x0",
            ]
        ]
        .set_index("anchor_id")
    )

    anchor_df = meta_one.copy()
    for lag in LAGS:
        anchor_df[f"z_sim_lag{lag}"] = pivot_z[lag]
        anchor_df[f"h_sim_lag{lag}"] = pivot_h[lag]

    anchor_df["z_1_gt_3"] = (
        anchor_df["z_sim_lag1"] > anchor_df["z_sim_lag3"]
    )
    anchor_df["z_3_gt_6"] = (
        anchor_df["z_sim_lag3"] > anchor_df["z_sim_lag6"]
    )
    anchor_df["z_full_monotonic"] = (
        anchor_df["z_1_gt_3"] & anchor_df["z_3_gt_6"]
    )
    anchor_df["h_1_gt_3"] = (
        anchor_df["h_sim_lag1"] > anchor_df["h_sim_lag3"]
    )
    anchor_df["h_3_gt_6"] = (
        anchor_df["h_sim_lag3"] > anchor_df["h_sim_lag6"]
    )
    anchor_df["h_full_monotonic"] = (
        anchor_df["h_1_gt_3"] & anchor_df["h_3_gt_6"]
    )
    anchor_df = anchor_df.reset_index()

    summary_rows = []
    for lag in LAGS:
        g = avg[avg["lag_hours"] == lag]
        summary_rows.append({
            "lag_hours": lag,
            "n_anchors": len(g),
            "z_mean": g["z_similarity_direction_mean"].mean(),
            "z_median": g["z_similarity_direction_mean"].median(),
            "z_std": g["z_similarity_direction_mean"].std(ddof=1),
            "h_mean": g["h_similarity_direction_mean"].mean(),
            "h_median": g["h_similarity_direction_mean"].median(),
            "h_std": g["h_similarity_direction_mean"].std(ddof=1),
        })
    summary_df = pd.DataFrame(summary_rows)

    long_path = exp / f"timelag_similarity_long_fp32_{checkpoint_tag}_step{step}.csv"
    anchor_path = exp / f"timelag_similarity_anchor_fp32_{checkpoint_tag}_step{step}.csv"
    summary_path = exp / f"timelag_similarity_summary_fp32_{checkpoint_tag}_step{step}.csv"

    long_df.to_csv(long_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_dir = exp / "diagnostic_plots" / f"timelag_gradation_fp32_{checkpoint_tag}_step{step}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    boxplot(
        avg.rename(columns={
            "z_similarity_direction_mean": "score"
        }),
        "score",
        plot_dir / f"timelag_similarity_z_boxplot_fp32_step{step}.png",
        "Projection z: same-valid-time similarity versus time lag",
    )
    boxplot(
        avg.rename(columns={
            "h_similarity_direction_mean": "score"
        }),
        "score",
        plot_dir / f"timelag_similarity_h_boxplot_fp32_step{step}.png",
        "Backbone h: same-valid-time similarity versus time lag",
    )

    print()
    print(summary_df.to_string(index=False))
    print()
    print(
        "z ordering fractions: "
        f"S1>S3={100*anchor_df['z_1_gt_3'].mean():.1f}% | "
        f"S3>S6={100*anchor_df['z_3_gt_6'].mean():.1f}% | "
        f"S1>S3>S6={100*anchor_df['z_full_monotonic'].mean():.1f}%"
    )
    print(
        "h ordering fractions: "
        f"S1>S3={100*anchor_df['h_1_gt_3'].mean():.1f}% | "
        f"S3>S6={100*anchor_df['h_3_gt_6'].mean():.1f}% | "
        f"S1>S3>S6={100*anchor_df['h_full_monotonic'].mean():.1f}%"
    )
    print()
    print(f"Long pairs : {long_path}")
    print(f"Per-anchor : {anchor_path}")
    print(f"Summary    : {summary_path}")
    print(f"Plots      : {plot_dir}")


if __name__ == "__main__":
    main()

