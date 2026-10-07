#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    ContrastiveIndex,
    NegativeRules,
    ContrastiveBatchSampler,
    HRRRArchive,
    ContrastiveDataset,
    PrecipTransform,
    contrastive_collate,
)


def default_paths():
    data = Path(os.environ["DATA_DIR"])
    out = data / "derived/benchmarks/dataloader_io_v1"
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": out,
    }


def parse_workers(text):
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    vals = list(dict.fromkeys(vals))
    if not vals or any(v < 0 for v in vals):
        raise ValueError("worker counts must be nonnegative integers")
    return vals


def check_batch(batch, batch_size, negative_count, patch_size):
    b = batch["anchor"].shape[0]
    expected = {
        "anchor": (b, 1, patch_size, patch_size),
        "positive": (b, 1, patch_size, patch_size),
        "negatives": (b, negative_count, 1, patch_size, patch_size),
        "negative_mask": (b, negative_count),
    }
    if b != batch_size:
        raise RuntimeError(f"Unexpected batch size {b}; expected {batch_size}")
    for key, shape in expected.items():
        if tuple(batch[key].shape) != shape:
            raise RuntimeError(f"{key} shape {tuple(batch[key].shape)} != {shape}")
    for key in ("anchor", "positive", "negatives"):
        if batch[key].dtype != torch.float32:
            raise RuntimeError(f"{key} dtype {batch[key].dtype}; expected float32")
        if not torch.isfinite(batch[key]).all():
            raise RuntimeError(f"Non-finite tensor found in {key}")


def shutdown_iterator(iterator):
    fn = getattr(iterator, "_shutdown_workers", None)
    if callable(fn):
        fn()


def benchmark_one(
    *,
    index,
    rules,
    manifest,
    transform,
    workers,
    repeat,
    epoch,
    batch_size,
    patch_size,
    negative_count,
    warmup_batches,
    timed_batches,
    prefetch_factor,
    pin_memory,
    seed,
):
    archive = HRRRArchive(manifest)
    sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=batch_size,
        patch_size=patch_size,
        negative_count=negative_count,
        negative_rules=rules,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    dataset = ContrastiveDataset(
        archive,
        negative_count=negative_count,
        transform=transform,
    )

    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=contrastive_collate,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**kwargs)
    iterator = iter(loader)

    startup_seconds = np.nan
    n_samples = 0
    n_negative_slots = 0
    n_masked = 0

    try:
        warmup_start = time.perf_counter()
        for i in range(warmup_batches):
            t0 = time.perf_counter()
            batch = next(iterator)
            dt = time.perf_counter() - t0
            if i == 0:
                startup_seconds = dt
                check_batch(batch, batch_size, negative_count, patch_size)
            _ = int(batch["negative_mask"].sum())
        warmup_seconds = time.perf_counter() - warmup_start

        timed_start = time.perf_counter()
        for i in range(timed_batches):
            batch = next(iterator)
            if i == 0:
                check_batch(batch, batch_size, negative_count, patch_size)

            b = int(batch["anchor"].shape[0])
            n_samples += b
            n_negative_slots += b * negative_count
            n_masked += int((~batch["negative_mask"]).sum())

            # Force access to the returned tensors in the main process.
            _ = (
                float(batch["anchor"].sum())
                + float(batch["positive"].sum())
                + float(batch["negatives"].sum())
            )

        timed_seconds = time.perf_counter() - timed_start

    finally:
        shutdown_iterator(iterator)
        archive.close()
        del iterator, loader, dataset, archive
        gc.collect()

    physical_patches = n_samples * (2 + negative_count)
    materialized_bytes = physical_patches * patch_size * patch_size * 4

    return {
        "repeat": repeat,
        "epoch": epoch,
        "num_workers": workers,
        "batch_size": batch_size,
        "negative_count": negative_count,
        "patch_size": patch_size,
        "warmup_batches": warmup_batches,
        "timed_batches": timed_batches,
        "prefetch_factor": prefetch_factor if workers > 0 else np.nan,
        "pin_memory": bool(pin_memory),
        "startup_first_batch_seconds": float(startup_seconds),
        "warmup_total_seconds": float(warmup_seconds),
        "timed_seconds": float(timed_seconds),
        "seconds_per_batch": float(timed_seconds / timed_batches),
        "batches_per_second": float(timed_batches / timed_seconds),
        "samples": int(n_samples),
        "samples_per_second": float(n_samples / timed_seconds),
        "physical_patches": int(physical_patches),
        "patches_per_second": float(physical_patches / timed_seconds),
        "materialized_mib_per_second": float(
            materialized_bytes / timed_seconds / (1024 ** 2)
        ),
        "negative_slots": int(n_negative_slots),
        "masked_negative_slots": int(n_masked),
        "masked_negative_fraction": float(
            n_masked / n_negative_slots if n_negative_slots else np.nan
        ),
    }


def main():
    d = default_paths()

    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--output-dir", default=str(d["output_dir"]))

    p.add_argument("--workers", default="0,2,4,8")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--timed-batches", type=int, default=20)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)

    p.add_argument("--negative-mode", choices=["climo45", "calendar_any"],
                   default="climo45")
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)

    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--epoch-base", type=int, default=100)
    args = p.parse_args()

    workers = parse_workers(args.workers)
    if args.repeats < 1 or args.warmup_batches < 1 or args.timed_batches < 1:
        raise SystemExit("repeats/warmup/timed batch counts must be >= 1")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HRRR CONTRASTIVE DATALOADER I/O BENCHMARK")
    print("=" * 80)
    print(f"Workers             : {workers}")
    print(f"Repeats             : {args.repeats}")
    print(f"Warmup batches      : {args.warmup_batches}")
    print(f"Timed batches       : {args.timed_batches}")
    print(f"Batch size          : {args.batch_size}")
    print(f"K negatives         : {args.negative_count}")
    print(f"Patches / sample    : {2 + args.negative_count}")
    print(f"Patch size          : {args.patch_size}")
    print(f"Negative mode       : {args.negative_mode}")
    print(f"Transform           : log1p_zscore")
    print(f"Prefetch factor     : {args.prefetch_factor}")
    print(f"Pin memory          : {args.pin_memory}")
    print(f"Output              : {outdir}")
    print()

    print("Loading metadata index...")
    t0 = time.perf_counter()
    index = ContrastiveIndex(args.anchors, args.negative_pool, args.manifest)
    rules = NegativeRules(
        mode=args.negative_mode,
        min_days=args.negative_min_days,
        climo_window_days=args.climo_window_days,
    )
    transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    index_seconds = time.perf_counter() - t0
    print(
        f"Metadata index ready in {index_seconds:.2f} s; "
        f"training anchors={len(index.anchors_for_split('train')):,}"
    )
    print()

    rows = []

    for repeat in range(args.repeats):
        epoch = args.epoch_base + repeat
        run_order = workers if repeat % 2 == 0 else list(reversed(workers))

        print("-" * 80)
        print(
            f"REPEAT {repeat + 1}/{args.repeats} | "
            f"sampler epoch={epoch} | order={run_order}"
        )
        print("-" * 80)

        for nw in run_order:
            print(f"num_workers={nw}: starting...", flush=True)
            row = benchmark_one(
                index=index,
                rules=rules,
                manifest=args.manifest,
                transform=transform,
                workers=nw,
                repeat=repeat,
                epoch=epoch,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                negative_count=args.negative_count,
                warmup_batches=args.warmup_batches,
                timed_batches=args.timed_batches,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                seed=args.seed,
            )
            rows.append(row)
            print(
                f"  first batch={row['startup_first_batch_seconds']:.2f}s | "
                f"steady={row['seconds_per_batch']:.2f}s/batch | "
                f"{row['samples_per_second']:.2f} samples/s | "
                f"{row['patches_per_second']:.1f} patches/s | "
                f"masked={100*row['masked_negative_fraction']:.2f}%"
            )
            time.sleep(1.0)
        print()

    results = pd.DataFrame(rows)
    csv_path = outdir / "dataloader_io_results.csv"
    results.to_csv(csv_path, index=False)

    summary_rows = []
    for nw, g in results.groupby("num_workers", sort=True):
        summary_rows.append(
            {
                "num_workers": int(nw),
                "runs": int(len(g)),
                "median_startup_first_batch_seconds": float(
                    g["startup_first_batch_seconds"].median()
                ),
                "median_seconds_per_batch": float(
                    g["seconds_per_batch"].median()
                ),
                "median_samples_per_second": float(
                    g["samples_per_second"].median()
                ),
                "min_samples_per_second": float(
                    g["samples_per_second"].min()
                ),
                "max_samples_per_second": float(
                    g["samples_per_second"].max()
                ),
                "median_patches_per_second": float(
                    g["patches_per_second"].median()
                ),
                "median_materialized_mib_per_second": float(
                    g["materialized_mib_per_second"].median()
                ),
                "masked_negative_fraction": float(
                    g["masked_negative_slots"].sum()
                    / g["negative_slots"].sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("num_workers")
    base = summary_df.loc[
        summary_df["num_workers"] == 0, "median_samples_per_second"
    ]
    baseline = float(base.iloc[0]) if len(base) else np.nan
    summary_df["speedup_vs_workers0"] = (
        summary_df["median_samples_per_second"] / baseline
        if np.isfinite(baseline) and baseline > 0
        else np.nan
    )

    best = summary_df.loc[summary_df["median_samples_per_second"].idxmax()]

    summary = {
        "version": 1,
        "purpose": "DataLoader/input-pipeline benchmark; no model/GPU",
        "configuration": {
            "workers": workers,
            "repeats": args.repeats,
            "warmup_batches": args.warmup_batches,
            "timed_batches": args.timed_batches,
            "batch_size": args.batch_size,
            "negative_count": args.negative_count,
            "patches_per_sample": 2 + args.negative_count,
            "patch_size": args.patch_size,
            "negative_mode": args.negative_mode,
            "transform": "log1p_zscore",
            "prefetch_factor": args.prefetch_factor,
            "pin_memory": bool(args.pin_memory),
            "seed": args.seed,
            "epoch_base": args.epoch_base,
        },
        "metadata_index_seconds": index_seconds,
        "summary_by_workers": summary_df.to_dict(orient="records"),
        "best_median_worker_count": int(best["num_workers"]),
        "best_median_samples_per_second": float(best["median_samples_per_second"]),
        "results_csv": str(csv_path),
    }

    json_path = outdir / "dataloader_io_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 80)
    print("SUMMARY (steady-state median across repeats)")
    print("=" * 80)
    print(
        summary_df[
            [
                "num_workers",
                "median_seconds_per_batch",
                "median_samples_per_second",
                "median_patches_per_second",
                "speedup_vs_workers0",
            ]
        ].to_string(
            index=False,
            formatters={
                "median_seconds_per_batch": "{:.3f}".format,
                "median_samples_per_second": "{:.3f}".format,
                "median_patches_per_second": "{:.1f}".format,
                "speedup_vs_workers0": "{:.2f}x".format,
            },
        )
    )
    print()
    print(
        f"Best median throughput: num_workers={int(best['num_workers'])}, "
        f"{best['median_samples_per_second']:.3f} samples/s"
    )
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

