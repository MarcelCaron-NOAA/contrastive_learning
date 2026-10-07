#!/usr/bin/env python3
"""
Evaluate learned A/P and A/N similarities on the fixed validation recipes.

Writes:
  validation_similarity_pairs_stepXXXX.csv
      one row per A/P pair and per valid A/N pair

  validation_similarity_anchor_summary_stepXXXX.csv
      one row per anchor with A/P similarity, mean A/N similarity,
      hardest-negative similarity, margin, retrieval result, and precip regime

This is a diagnostic on the already-selected validation checkpoint; it is not
a sealed test-set evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import HRRRArchive, PrecipTransform, ContrastiveRecipe
from src.models.precip_resnet import PrecipResNet18


GRID_SHAPE = (1059, 1799)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/first_real_train_validation_v1"
    return {
        "experiment_dir": exp,
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def load_recipes(path: Path):
    recipes = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["negative_ids"] = tuple(d["negative_ids"])
            recipes.append(ContrastiveRecipe(**d))
    if not recipes:
        raise RuntimeError(f"No recipes found in {path}")
    return recipes


def array_equal(a, b):
    if a.shape != b.shape:
        return False
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(same))


def season_from_month(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def precip_stats(raw, wet_threshold):
    x = np.asarray(raw, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float32)
    x = np.maximum(x, 0.0)

    wet = x >= wet_threshold
    wet_values = x[wet]

    return {
        "mean_mm": float(np.mean(x)),
        "max_mm": float(np.max(x)),
        "p95_mm": float(np.quantile(x, 0.95)),
        "p99_mm": float(np.quantile(x, 0.99)),
        "wet_fraction": float(np.mean(wet)),
        "wet_pixel_p90_mm": (
            float(np.quantile(wet_values, 0.90))
            if wet_values.size
            else 0.0
        ),
        "wet_pixel_count": int(wet_values.size),
    }


def intensity_bin(v):
    if not np.isfinite(v) or v < 0.1:
        return "dry"
    if v < 1.0:
        return "0.1-1"
    if v < 5.0:
        return "1-5"
    if v < 10.0:
        return "5-10"
    if v < 25.0:
        return "10-25"
    if v < 50.0:
        return "25-50"
    return ">=50"


class FixedSimilarityDataset(Dataset):
    def __init__(
        self,
        recipes,
        manifest,
        transform,
        wet_threshold=0.1,
    ):
        self.recipes = recipes
        self.archive = HRRRArchive(manifest)
        self.transform = transform
        self.wet_threshold = float(wet_threshold)

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, i):
        r = self.recipes[i]

        a_raw = self.archive.read_patch(
            r.anchor_id, r.y0, r.x0, r.patch_size
        )
        p_raw = self.archive.read_patch(
            r.positive_id, r.y0, r.x0, r.patch_size
        )

        n_raw_list = []
        n_tensors = []
        n_mask = []

        for nid in r.negative_ids:
            n_raw = self.archive.read_patch(
                nid, r.y0, r.x0, r.patch_size
            )
            valid = not (
                array_equal(n_raw, a_raw)
                or array_equal(n_raw, p_raw)
            )
            n_raw_list.append(n_raw)
            n_tensors.append(self.transform(n_raw))
            n_mask.append(valid)

        ast = precip_stats(a_raw, self.wet_threshold)
        pst = precip_stats(p_raw, self.wet_threshold)

        ts = pd.Timestamp(r.anchor_valid_time)
        meta = {
            "anchor_id": r.anchor_id,
            "positive_id": r.positive_id,
            "positive_direction": r.positive_direction,
            "split": r.split,
            "anchor_valid_time": r.anchor_valid_time,
            "anchor_forecast_hour": int(r.anchor_forecast_hour),
            "anchor_valid_hour_utc": int(r.anchor_valid_hour_utc),
            "season": season_from_month(ts.month),
            "crop_y0": int(r.y0),
            "crop_x0": int(r.x0),
            "patch_size": int(r.patch_size),
            "negative_ids": tuple(r.negative_ids),
        }

        for key, val in ast.items():
            meta[f"anchor_{key}"] = val
        for key, val in pst.items():
            meta[f"positive_{key}"] = val

        meta["wet_dry"] = (
            "wet" if ast["wet_pixel_count"] > 0 else "dry"
        )
        meta["intensity_bin"] = intensity_bin(
            ast["wet_pixel_p90_mm"]
        )

        return {
            "anchor": self.transform(a_raw),
            "positive": self.transform(p_raw),
            "negatives": torch.stack(n_tensors, dim=0),
            "negative_mask": torch.tensor(n_mask, dtype=torch.bool),
            "metadata": meta,
        }


def collate(items):
    return {
        "anchor": torch.stack([x["anchor"] for x in items]),
        "positive": torch.stack([x["positive"] for x in items]),
        "negatives": torch.stack([x["negatives"] for x in items]),
        "negative_mask": torch.stack([x["negative_mask"] for x in items]),
        "metadata": [x["metadata"] for x in items],
    }


def encode(model, a, p, n):
    B, K = n.shape[:2]
    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])], dim=0
    )
    h, z = model(x)

    h_a = h[:B]
    h_p = h[B:2 * B]
    h_n = h[2 * B:].reshape(B, K, -1)

    z_a = z[:B]
    z_p = z[B:2 * B]
    z_n = z[2 * B:].reshape(B, K, -1)

    h_a = torch.nn.functional.normalize(h_a, dim=-1)
    h_p = torch.nn.functional.normalize(h_p, dim=-1)
    h_n = torch.nn.functional.normalize(h_n, dim=-1)

    return (h_a, h_p, h_n), (z_a, z_p, z_n)


def main():
    d = defaults()

    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", default=str(d["experiment_dir"]))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--wet-threshold-mm", type=float, default=0.1)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else exp / "checkpoint_best.pt"
    )
    recipe_path = exp / "fixed_validation_recipes.jsonl"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    elif args.allow_cpu:
        device = torch.device("cpu")
        amp = False
    else:
        raise SystemExit(
            "CUDA not available. Run on GPU or pass --allow-cpu."
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    step = int(checkpoint.get("step", -1))

    model_spec = checkpoint.get("model", {})
    model = PrecipResNet18(
        projection_hidden_dim=int(
            model_spec.get("projection_hidden_dim", 512)
        ),
        projection_dim=int(model_spec.get("projection_dim", 128)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )
    recipes = load_recipes(recipe_path)

    dataset = FixedSimilarityDataset(
        recipes,
        args.manifest,
        transform,
        wet_threshold=args.wet_threshold_mm,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    pair_rows = []
    anchor_rows = []

    print("=" * 82)
    print("FIXED VALIDATION SIMILARITY DECOMPOSITION")
    print("=" * 82)
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Checkpoint step : {step}")
    print(f"Recipes         : {len(recipes)}")
    print(f"Device          : {device}")
    print()

    with torch.no_grad():
        for ib, batch in enumerate(loader, start=1):
            a = batch["anchor"].to(device, non_blocking=True)
            p_ = batch["positive"].to(device, non_blocking=True)
            n = batch["negatives"].to(device, non_blocking=True)
            mask = batch["negative_mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                (ha, hp, hn), (za, zp, zn) = encode(
                    model, a, p_, n
                )

                z_ap = torch.sum(za * zp, dim=-1)
                z_an = torch.einsum("bd,bkd->bk", za, zn)

                h_ap = torch.sum(ha * hp, dim=-1)
                h_an = torch.einsum("bd,bkd->bk", ha, hn)

            z_ap = z_ap.float().cpu().numpy()
            z_an = z_an.float().cpu().numpy()
            h_ap = h_ap.float().cpu().numpy()
            h_an = h_an.float().cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)

            for i, meta in enumerate(batch["metadata"]):
                base = {
                    k: v for k, v in meta.items()
                    if k != "negative_ids"
                }

                pair_rows.append({
                    **base,
                    "checkpoint_step": step,
                    "pair_type": "positive",
                    "pair_id": meta["positive_id"],
                    "negative_slot": np.nan,
                    "z_similarity": float(z_ap[i]),
                    "h_similarity": float(h_ap[i]),
                })

                valid_z = []
                valid_h = []
                valid_ids = []

                for k, nid in enumerate(meta["negative_ids"]):
                    if not mask_np[i, k]:
                        continue

                    zval = float(z_an[i, k])
                    hval = float(h_an[i, k])

                    valid_z.append(zval)
                    valid_h.append(hval)
                    valid_ids.append(str(nid))

                    pair_rows.append({
                        **base,
                        "checkpoint_step": step,
                        "pair_type": "negative",
                        "pair_id": str(nid),
                        "negative_slot": int(k),
                        "z_similarity": zval,
                        "h_similarity": hval,
                    })

                if valid_z:
                    z_arr = np.asarray(valid_z)
                    h_arr = np.asarray(valid_h)
                    zhard = int(np.argmax(z_arr))
                    hhard = int(np.argmax(h_arr))

                    anchor_rows.append({
                        **base,
                        "checkpoint_step": step,
                        "z_cosine_ap": float(z_ap[i]),
                        "z_cosine_an_mean": float(np.mean(z_arr)),
                        "z_cosine_an_max": float(np.max(z_arr)),
                        "z_margin": float(z_ap[i] - np.max(z_arr)),
                        "z_top1": int(z_ap[i] > np.max(z_arr)),
                        "z_hardest_negative_id": valid_ids[zhard],
                        "h_cosine_ap": float(h_ap[i]),
                        "h_cosine_an_mean": float(np.mean(h_arr)),
                        "h_cosine_an_max": float(np.max(h_arr)),
                        "h_margin": float(h_ap[i] - np.max(h_arr)),
                        "h_top1": int(h_ap[i] > np.max(h_arr)),
                        "h_hardest_negative_id": valid_ids[hhard],
                        "valid_negative_count": len(valid_z),
                    })

            if ib == 1 or ib % 16 == 0 or ib == len(loader):
                print(
                    f"Evaluated batch {ib:3d}/{len(loader)} "
                    f"({min(ib*args.batch_size, len(recipes)):4d} anchors)"
                )

    pair_df = pd.DataFrame(pair_rows)
    anchor_df = pd.DataFrame(anchor_rows)

    pair_path = exp / f"validation_similarity_pairs_step{step}.csv"
    anchor_path = (
        exp / f"validation_similarity_anchor_summary_step{step}.csv"
    )

    pair_df.to_csv(pair_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)

    print()
    print(f"Wrote pair scores   : {pair_path}")
    print(f"Wrote anchor summary: {anchor_path}")
    print()
    print(
        "z mean similarities: "
        f"AP={anchor_df['z_cosine_ap'].mean():.4f}, "
        f"ANmean={anchor_df['z_cosine_an_mean'].mean():.4f}, "
        f"ANmax={anchor_df['z_cosine_an_max'].mean():.4f}, "
        f"margin={anchor_df['z_margin'].mean():+.4f}"
    )


if __name__ == "__main__":
    main()

