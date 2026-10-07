#!/usr/bin/env python3
"""
First real HRRR contrastive train/validation experiment.

Purpose
-------
Demonstrate GENERALIZATION beyond the tiny cached overfit test, while keeping
the experiment intentionally modest.

Training:
  * train split only
  * dynamic climo45 negatives
  * each anchor visited at most once in this short run
  * random +/-1-h positive direction
  * random 256x256 crop
  * 8 explicit negatives
  * log1p + training-derived z-score
  * 1-channel ResNet-18 with GroupNorm(32 groups)
  * symmetric explicit-negative InfoNCE

Validation:
  * held-out validation split only
  * a fixed deterministic set of recipes cached once at startup
  * calendar_any negatives, NOT climo45

Why calendar_any for validation?
The validation period is only about one year long. Requiring a different year plus
a +/-45-day climatological window would leave many validation anchors with no
eligible N population and would bias evaluation toward the split boundaries.
calendar_any gives a geographically/temporally held-out, reproducible
validation task using the same lead/hour/>90-day rules. Every future training
ablation should use the exact same fixed validation construction.

This is still an internal self-supervised validation experiment. Stage-IV
forecast/observation verification comes later.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from src.models.precip_resnet_gn import PrecipResNet18GN
from src.training.contrastive_loss import SymmetricExplicitInfoNCE


def defaults():
    data = Path(os.environ["DATA_DIR"])
    out = data / "derived/experiments/groupnorm_pilot_v1"
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": out,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shutdown_iterator(iterator):
    fn = getattr(iterator, "_shutdown_workers", None)
    if callable(fn):
        fn()


def loader_kwargs(dataset, sampler, workers, pin_memory, prefetch_factor):
    kw = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=contrastive_collate,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return kw


def move_core_batch(batch, device):
    return {
        "anchor": batch["anchor"].to(device, non_blocking=True),
        "positive": batch["positive"].to(device, non_blocking=True),
        "negatives": batch["negatives"].to(device, non_blocking=True),
        "negative_mask": batch["negative_mask"].to(
            device, non_blocking=True
        ),
    }


def encode_batch(model, batch):
    a = batch["anchor"]
    p = batch["positive"]
    n = batch["negatives"]

    B, K = n.shape[:2]
    x = torch.cat(
        [a, p, n.reshape(B * K, *n.shape[2:])],
        dim=0,
    )

    h, z = model(x)

    h_a = h[:B]
    h_p = h[B:2 * B]
    h_n = h[2 * B:].reshape(B, K, -1)

    z_a = z[:B]
    z_p = z[B:2 * B]
    z_n = z[2 * B:].reshape(B, K, -1)

    return (h_a, h_p, h_n), (z_a, z_p, z_n)


def metric_to_float(v):
    if torch.is_tensor(v):
        return float(v.detach().float().cpu())
    return float(v)


class WeightedMetricAccumulator:
    def __init__(self):
        self.total_weight = 0.0
        self.sums = {}

    def update(self, metrics):
        weight = metric_to_float(metrics["active_query_count"])
        if weight <= 0:
            return

        self.total_weight += weight
        for key, value in metrics.items():
            if key in ("active_query_count",):
                continue
            val = metric_to_float(value)
            if np.isfinite(val):
                self.sums[key] = self.sums.get(key, 0.0) + weight * val

    def result(self):
        if self.total_weight <= 0:
            return {}
        out = {
            k: v / self.total_weight
            for k, v in self.sums.items()
        }
        out["active_query_count"] = self.total_weight
        return out


@torch.no_grad()
def evaluate_cached(
    model,
    criterion,
    cached_validation,
    device,
    use_amp,
):
    model.eval()

    z_acc = WeightedMetricAccumulator()
    h_acc = WeightedMetricAccumulator()

    for cpu_batch in cached_validation:
        batch = move_core_batch(cpu_batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            (h_a, h_p, h_n), (z_a, z_p, z_n) = encode_batch(
                model, batch
            )

            _, z_metrics = criterion(
                z_a,
                z_p,
                z_n,
                batch["negative_mask"],
            )

            # The backbone representation is not the optimized projection
            # space, but we track normalized h cosine geometry because h may
            # ultimately be useful for the forecast-verification metric.
            hn_a = F.normalize(h_a, dim=-1)
            hn_p = F.normalize(h_p, dim=-1)
            hn_n = F.normalize(h_n, dim=-1)

            _, h_metrics = criterion(
                hn_a,
                hn_p,
                hn_n,
                batch["negative_mask"],
            )

        z_acc.update(z_metrics)
        h_acc.update(h_metrics)

    return {
        "z": z_acc.result(),
        "h": h_acc.result(),
    }


def print_validation(step, metrics):
    z = metrics["z"]
    h = metrics["h"]
    print(
        f"VAL step={step:5d} | "
        f"z_loss={z['loss']:.4f} "
        f"z_AP={z['cosine_ap']:.4f} "
        f"z_AN={z['cosine_an']:.4f} "
        f"z_margin={z['margin_a']:+.4f} "
        f"z_top1={z['top1_a_pct']:.1f}% | "
        f"h_AP={h['cosine_ap']:.4f} "
        f"h_AN={h['cosine_an']:.4f} "
        f"h_margin={h['margin_a']:+.4f} "
        f"h_top1={h['top1_a_pct']:.1f}%"
    )


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    step,
    args,
    best_val_loss,
):
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_z_loss": float(best_val_loss),
            "model": {
                "name": "PrecipResNet18GN",
                "normalization": "GroupNorm32",
                "projection_hidden_dim": args.projection_hidden_dim,
                "projection_dim": args.projection_dim,
            },
            "criterion": {
                "name": "SymmetricExplicitInfoNCE",
                "temperature": args.temperature,
            },
            "preprocessing": {
                "transform": "log1p_zscore",
                "stats_json": str(args.stats_json),
            },
            "training_sampling": {
                "split": "train",
                "negative_mode": "climo45",
                "negative_count": args.negative_count,
                "negative_min_days": args.negative_min_days,
                "climo_window_days": args.climo_window_days,
                "patch_size": args.patch_size,
            },
            "validation_sampling": {
                "split": args.validation_split,
                "negative_mode": "calendar_any",
                "negative_count": args.negative_count,
                "negative_min_days": args.negative_min_days,
                "patch_size": args.patch_size,
                "fixed_validation_samples": args.validation_samples,
                "validation_seed": args.validation_seed,
            },
            "learning_rate_schedule": {
                "type": "linear_warmup_then_cosine_decay_no_restarts",
                "warmup_start_lr": args.warmup_start_lr,
                "peak_lr": args.learning_rate,
                "min_lr": args.min_learning_rate,
                "warmup_fraction": args.warmup_fraction,
            },
            "seed": args.seed,
        },
        path,
    )



class SkipBatchSampler:
    """Skip already-completed deterministic batches without materializing data."""
    def __init__(self, base_sampler, skip_batches=0):
        self.base_sampler = base_sampler
        self.skip_batches = int(skip_batches)

    def __iter__(self):
        import itertools
        return itertools.islice(iter(self.base_sampler), self.skip_batches, None)

    def __len__(self):
        return max(0, len(self.base_sampler) - self.skip_batches)


def learning_rate_for_step(
    step,
    total_steps,
    warmup_steps,
    warmup_start_lr,
    peak_lr,
    min_lr,
):
    """Linear warmup followed by cosine decay without restarts."""
    step = int(step)
    if step <= warmup_steps:
        if warmup_steps <= 1:
            return float(peak_lr)
        frac = (step - 1) / float(warmup_steps - 1)
        return float(warmup_start_lr + frac * (peak_lr - warmup_start_lr))

    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / float(denom)
    progress = min(1.0, max(0.0, progress))
    return float(
        min_lr
        + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    )


def write_histories(outdir, train_rows, validation_rows):
    train_csv = Path(outdir) / "training_history.csv"
    if train_rows:
        with train_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(train_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(train_rows)

    val_csv = Path(outdir) / "validation_history.csv"
    if validation_rows:
        with val_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(validation_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(validation_rows)

    return train_csv, val_csv

def main():
    d = defaults()

    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--output-dir", default=str(d["output_dir"]))

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--validation-split", default="validation")

    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.10)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-start-lr", type=float, default=3e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-learning-rate", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--resume-checkpoint", default=None)

    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--validation-seed", type=int, default=20261001)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    if args.epochs != 1:
        raise SystemExit("This pilot uses epoch=0 sampling only; keep --epochs 1.")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if not (0.0 <= args.warmup_fraction < 1.0):
        raise SystemExit("--warmup-fraction must be in [0,1).")
    if not (0.0 <= args.min_learning_rate <= args.learning_rate):
        raise SystemExit("--min-learning-rate must be between 0 and peak LR.")
    if not (0.0 <= args.warmup_start_lr <= args.learning_rate):
        raise SystemExit("--warmup-start-lr must be between 0 and peak LR.")
    if args.validate_every < 1:
        raise SystemExit("--validate-every must be >= 1")
    if args.validation_samples < args.batch_size:
        raise SystemExit("--validation-samples is too small")
    if args.validation_samples % args.batch_size != 0:
        raise SystemExit(
            "--validation-samples must be divisible by --batch-size "
            "for this v1 run."
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this training experiment.")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    use_amp = not args.no_amp

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = outdir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2) + "\n")

    print("=" * 86)
    print("GROUPNORM HRRR CONTRASTIVE TRAIN / VALIDATION PILOT")
    print("=" * 86)
    print(f"GPU                  : {torch.cuda.get_device_name(0)}")
    print(f"AMP                  : {'bf16' if use_amp else 'off'}")
    print(f"Training extent      : first {args.max_steps:,} steps of epoch-0 anchor order")
    print("Normalization        : GroupNorm(32 groups); no running statistics")
    print(f"Logical batch size   : {args.batch_size}")
    print(f"K negatives          : {args.negative_count}")
    print(
        f"Physical images/step : "
        f"{args.batch_size * (args.negative_count + 2)}"
    )
    print(f"DataLoader workers   : {args.num_workers}")
    print(f"Prefetch factor      : {args.prefetch_factor}")
    print(f"Train negatives      : climo45")
    print(f"Validation negatives : calendar_any (fixed)")
    print(f"Validation split     : {args.validation_split}")
    print(f"Validation samples   : {args.validation_samples}")
    print(f"Validation cadence   : every {args.validate_every} steps")
    print(f"Peak learning rate   : {args.learning_rate}")
    print(f"Warmup start LR      : {args.warmup_start_lr}")
    print(f"Warmup fraction      : {args.warmup_fraction:.3f}")
    print(f"Minimum LR           : {args.min_learning_rate}")
    print(f"Temperature          : {args.temperature}")
    print(f"Output               : {outdir}")
    print()

    # ------------------------------------------------------------------
    # Shared metadata + transform
    # ------------------------------------------------------------------
    print("Loading metadata index...")
    t_index = time.perf_counter()
    index = ContrastiveIndex(
        args.anchors,
        args.negative_pool,
        args.manifest,
    )
    transform = PrecipTransform.from_stats_json(
        "log1p_zscore",
        args.stats_json,
    )
    print(
        f"Metadata ready in {time.perf_counter()-t_index:.2f}s | "
        f"train={len(index.anchors_for_split('train')):,} | "
        f"{args.validation_split}={len(index.anchors_for_split(args.validation_split)):,}"
    )
    print()

    # ------------------------------------------------------------------
    # Fixed held-out validation cache
    # ------------------------------------------------------------------
    print("-" * 86)
    print(f"Building FIXED validation cache from split={args.validation_split!r}")
    print("-" * 86)

    val_rules = NegativeRules(
        mode="calendar_any",
        min_days=args.negative_min_days,
        climo_window_days=args.climo_window_days,
    )
    val_sampler = ContrastiveBatchSampler(
        index,
        split=args.validation_split,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=val_rules,
        seed=args.validation_seed,
    )
    val_sampler.set_epoch(0)

    val_archive = HRRRArchive(args.manifest)
    val_dataset = ContrastiveDataset(
        val_archive,
        negative_count=args.negative_count,
        transform=transform,
    )
    val_loader = DataLoader(
        **loader_kwargs(
            val_dataset,
            val_sampler,
            args.num_workers,
            True,
            args.prefetch_factor,
        )
    )

    val_iter = iter(val_loader)
    n_val_batches = args.validation_samples // args.batch_size
    cached_validation = []
    validation_recipes = []

    t_val_cache = time.perf_counter()
    try:
        for i in range(n_val_batches):
            batch = next(val_iter)

            cached_validation.append(
                {
                    "anchor": batch["anchor"].clone(),
                    "positive": batch["positive"].clone(),
                    "negatives": batch["negatives"].clone(),
                    "negative_mask": batch["negative_mask"].clone(),
                }
            )
            validation_recipes.extend(batch["metadata"])

            if (
                i == 0
                or (i + 1) % 16 == 0
                or i + 1 == n_val_batches
            ):
                print(
                    f"Cached validation batch {i+1:3d}/{n_val_batches} "
                    f"({(i+1)*args.batch_size:4d} samples)"
                )
    finally:
        shutdown_iterator(val_iter)
        val_archive.close()
        del val_iter, val_loader, val_dataset, val_archive

    val_cache_seconds = time.perf_counter() - t_val_cache
    print(
        f"Validation cache complete: {args.validation_samples} samples "
        f"in {val_cache_seconds:.1f}s"
    )

    recipe_path = outdir / "fixed_validation_recipes.jsonl"
    with recipe_path.open("w") as f:
        for recipe in validation_recipes:
            f.write(json.dumps(recipe) + "\n")
    print(f"Fixed recipes saved: {recipe_path}")

    # Scientific-control check: this pilot is intended to use exactly the same
    # fixed validation recipes as the BatchNorm full-epoch baseline.
    baseline_recipe_path = (
        Path(os.environ["DATA_DIR"])
        / "derived/experiments/full_epoch_train_validation_v1"
        / "fixed_validation_recipes.jsonl"
    )
    if baseline_recipe_path.exists():
        same_recipes = (
            baseline_recipe_path.read_bytes()
            == recipe_path.read_bytes()
        )
        print(
            "Validation recipes match BN baseline exactly: "
            f"{same_recipes}"
        )
        if not same_recipes:
            raise RuntimeError(
                "GroupNorm validation recipes do not exactly match the "
                "BatchNorm baseline."
            )
    else:
        print(
            "WARNING: BN baseline fixed_validation_recipes.jsonl was not "
            "found, so exact recipe identity could not be verified."
        )
    print()

    # ------------------------------------------------------------------
    # Dynamic training DataLoader
    # ------------------------------------------------------------------
    train_rules = NegativeRules(
        mode="climo45",
        min_days=args.negative_min_days,
        climo_window_days=args.climo_window_days,
    )
    train_sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=train_rules,
        seed=args.seed,
    )
    train_sampler.set_epoch(0)

    full_epoch_steps = len(train_sampler)
    total_anchors = train_sampler.n_anchors
    run_steps = min(args.max_steps, full_epoch_steps)
    warmup_steps = max(
        1, int(round(full_epoch_steps * args.warmup_fraction))
    )

    print(
        f"Full anchor epoch    : {total_anchors:,} anchors in "
        f"{full_epoch_steps:,} minibatches"
    )
    print(f"Pilot steps          : {run_steps:,}")
    print(
        "LR schedule reference: full-epoch schedule "
        "(not compressed into the pilot)"
    )
    print(f"Warmup steps         : {warmup_steps:,}")
    print()

    train_archive = HRRRArchive(args.manifest)
    train_dataset = ContrastiveDataset(
        train_archive,
        negative_count=args.negative_count,
        transform=transform,
    )

    # ------------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------------
    model = PrecipResNet18GN(
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)
    criterion = SymmetricExplicitInfoNCE(
        temperature=args.temperature
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    resume_best_val_loss = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        resume_best_val_loss = float(ckpt.get("best_val_z_loss", float("inf")))
        if not (0 <= start_step < run_steps):
            raise RuntimeError(
                f"Resume checkpoint step={start_step} is incompatible with "
                f"pilot run_steps={run_steps}."
            )
        print(f"Resuming checkpoint   : {resume_path}")
        print(f"Resume after step     : {start_step:,}")

    effective_sampler = SkipBatchSampler(
        train_sampler, skip_batches=start_step
    )
    train_loader = DataLoader(
        **loader_kwargs(
            train_dataset,
            effective_sampler,
            args.num_workers,
            True,
            args.prefetch_factor,
        )
    )
    train_iter = iter(train_loader)

    n_params = sum(p.numel() for p in model.parameters())
    print("-" * 86)
    print(f"Model parameters: {n_params:,}")
    print("-" * 86)

    # ------------------------------------------------------------------
    # Baseline validation before any optimization
    # ------------------------------------------------------------------
    train_rows = []
    validation_rows = []

    # If resuming in the same output directory, preserve already-written history.
    train_csv_existing = outdir / "training_history.csv"
    val_csv_existing = outdir / "validation_history.csv"
    if start_step > 0 and train_csv_existing.exists():
        train_rows = pd.read_csv(train_csv_existing).to_dict("records")
    if start_step > 0 and val_csv_existing.exists():
        validation_rows = pd.read_csv(val_csv_existing).to_dict("records")

    initial_val = evaluate_cached(
        model,
        criterion,
        cached_validation,
        device,
        False,  # validation scoring in FP32
    )
    print_validation(start_step, initial_val)

    if not validation_rows or int(validation_rows[-1]["step"]) != start_step:
        validation_rows.append(
            {
                "step": start_step,
                **{f"z_{k}": v for k, v in initial_val["z"].items()},
                **{f"h_{k}": v for k, v in initial_val["h"].items()},
            }
        )

    if resume_best_val_loss is None:
        best_val_loss = initial_val["z"]["loss"]
        best_step = start_step
        save_checkpoint(
            outdir / "checkpoint_best.pt",
            model=model,
            optimizer=optimizer,
            step=start_step,
            args=args,
            best_val_loss=best_val_loss,
        )
    else:
        best_val_loss = resume_best_val_loss
        best_step = start_step

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    window = []
    run_start = time.perf_counter()
    data_wait_total = 0.0
    compute_total = 0.0

    try:
        for step in range(start_step + 1, run_steps + 1):
            current_lr = learning_rate_for_step(
                step,
                full_epoch_steps,
                warmup_steps,
                args.warmup_start_lr,
                args.learning_rate,
                args.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            wait_start = time.perf_counter()
            cpu_batch = next(train_iter)
            data_wait = time.perf_counter() - wait_start
            data_wait_total += data_wait

            batch = move_core_batch(cpu_batch, device)

            compute_start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                _, (z_a, z_p, z_n) = encode_batch(model, batch)
                loss, metrics = criterion(
                    z_a,
                    z_p,
                    z_n,
                    batch["negative_mask"],
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}: {float(loss)}"
                )

            active_count = metric_to_float(
                metrics["active_query_count"]
            )

            if active_count > 0:
                loss.backward()

                finite_grad = True
                sq_norm = 0.0
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        finite_grad = False
                        break
                    sq_norm += float(
                        torch.sum(
                            param.grad.detach().float() ** 2
                        ).cpu()
                    )

                if not finite_grad:
                    raise RuntimeError(
                        f"Non-finite gradient at step {step}"
                    )

                grad_norm = math.sqrt(sq_norm)
                optimizer.step()
            else:
                grad_norm = 0.0

            # Synchronize for honest GPU-compute timing.
            torch.cuda.synchronize()
            compute_time = time.perf_counter() - compute_start
            compute_total += compute_time

            row = {
                "step": step,
                "learning_rate": current_lr,
                "logical_batch_size": int(batch["anchor"].shape[0]),
                "loss": metric_to_float(metrics["loss"]),
                "cosine_ap": metric_to_float(metrics["cosine_ap"]),
                "cosine_an": metric_to_float(metrics["cosine_an"]),
                "cosine_pn": metric_to_float(metrics["cosine_pn"]),
                "margin_a": metric_to_float(metrics["margin_a"]),
                "margin_p": metric_to_float(metrics["margin_p"]),
                "top1_a_pct": metric_to_float(
                    metrics["top1_a_pct"]
                ),
                "top1_p_pct": metric_to_float(
                    metrics["top1_p_pct"]
                ),
                "valid_negative_fraction": metric_to_float(
                    metrics["valid_negative_fraction"]
                ),
                "active_query_fraction": metric_to_float(
                    metrics["active_query_fraction"]
                ),
                "active_query_count": active_count,
                "grad_norm": grad_norm,
                "data_wait_seconds": data_wait,
                "compute_seconds": compute_time,
            }
            train_rows.append(row)
            window.append(row)

            if step % args.print_every == 0 or step == 1:
                recent = window[-min(len(window), args.print_every):]
                def avg(key):
                    vals = [x[key] for x in recent if np.isfinite(x[key])]
                    return float(np.mean(vals)) if vals else np.nan

                print(
                    f"TRAIN step={step:5d}/{run_steps} | lr={current_lr:.3e} | "
                    f"loss={avg('loss'):.4f} "
                    f"AP={avg('cosine_ap'):.4f} "
                    f"AN={avg('cosine_an'):.4f} "
                    f"margin={avg('margin_a'):+.4f} "
                    f"top1={avg('top1_a_pct'):.1f}% | "
                    f"wait={avg('data_wait_seconds'):.3f}s "
                    f"gpu={avg('compute_seconds'):.3f}s"
                )

            if step % args.validate_every == 0:
                val = evaluate_cached(
                    model,
                    criterion,
                    cached_validation,
                    device,
                    False,  # validation scoring in FP32
                )
                print_validation(step, val)

                validation_rows.append(
                    {
                        "step": step,
                        **{
                            f"z_{k}": v
                            for k, v in val["z"].items()
                        },
                        **{
                            f"h_{k}": v
                            for k, v in val["h"].items()
                        },
                    }
                )

                if val["z"]["loss"] < best_val_loss:
                    best_val_loss = val["z"]["loss"]
                    best_step = step
                    save_checkpoint(
                        outdir / "checkpoint_best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        args=args,
                        best_val_loss=best_val_loss,
                    )
                    print(
                        f"  -> new best fixed-validation z loss "
                        f"{best_val_loss:.4f}"
                    )

                write_histories(outdir, train_rows, validation_rows)

            if step % args.checkpoint_every == 0 and step < run_steps:
                save_checkpoint(
                    outdir / f"checkpoint_step{step:05d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    best_val_loss=best_val_loss,
                )
                write_histories(outdir, train_rows, validation_rows)

        run_seconds = time.perf_counter() - run_start

    finally:
        shutdown_iterator(train_iter)
        train_archive.close()

    if not validation_rows or int(validation_rows[-1]["step"]) != run_steps:
        final_val = evaluate_cached(
            model,
            criterion,
            cached_validation,
            device,
            False,
        )
        print_validation(run_steps, final_val)
        validation_rows.append(
            {
                "step": run_steps,
                **{f"z_{k}": v for k, v in final_val["z"].items()},
                **{f"h_{k}": v for k, v in final_val["h"].items()},
            }
        )
        if final_val["z"]["loss"] < best_val_loss:
            best_val_loss = final_val["z"]["loss"]
            best_step = run_steps
            save_checkpoint(
                outdir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                step=run_steps,
                args=args,
                best_val_loss=best_val_loss,
            )

    # Final checkpoint regardless of best step.
    save_checkpoint(
        outdir / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        step=run_steps,
        args=args,
        best_val_loss=best_val_loss,
    )

    # ------------------------------------------------------------------
    # Write histories
    # ------------------------------------------------------------------
    train_csv, val_csv = write_histories(
        outdir, train_rows, validation_rows
    )

    initial_z = validation_rows[0]["z_loss"]
    final_z = validation_rows[-1]["z_loss"]

    summary = {
        "status": "complete",
        "purpose": (
            "controlled GroupNorm partial-epoch self-supervised train/validation experiment; "
            "not final forecast-verification evaluation"
        ),
        "train_steps": run_steps,
        "full_epoch_steps": full_epoch_steps,
        "logical_batch_size": args.batch_size,
        "training_logical_samples": int(sum(int(r["logical_batch_size"]) for r in train_rows)),
        "negative_count": args.negative_count,
        "workers": args.num_workers,
        "validation_samples": args.validation_samples,
        "validation_negative_mode": "calendar_any",
        "training_negative_mode": "climo45",
        "learning_rate_schedule": {
            "type": "linear_warmup_then_cosine_decay_no_restarts",
            "warmup_steps": warmup_steps,
            "warmup_fraction": args.warmup_fraction,
            "warmup_start_lr": args.warmup_start_lr,
            "peak_lr": args.learning_rate,
            "min_lr": args.min_learning_rate,
        },
        "initial_validation_z_loss": initial_z,
        "final_validation_z_loss": final_z,
        "best_validation_z_loss": best_val_loss,
        "best_step": best_step,
        "runtime_seconds": run_seconds,
        "mean_data_wait_seconds_per_step": (
            data_wait_total / max(1, run_steps - start_step)
        ),
        "mean_compute_seconds_per_step": (
            compute_total / max(1, run_steps - start_step)
        ),
        "logical_samples_per_second_end_to_end": (
            sum(int(r["logical_batch_size"]) for r in train_rows if int(r["step"]) > start_step) / run_seconds
        ),
        "files": {
            "config": str(config_path),
            "training_history": str(train_csv),
            "validation_history": str(val_csv),
            "fixed_validation_recipes": str(recipe_path),
            "checkpoint_best": str(
                outdir / "checkpoint_best.pt"
            ),
            "checkpoint_last": str(
                outdir / "checkpoint_last.pt"
            ),
        },
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 86)
    print("RUN COMPLETE")
    print("=" * 86)
    print(
        f"Validation z loss : "
        f"{initial_z:.4f} -> {final_z:.4f}"
    )
    print(
        f"Best validation   : "
        f"{best_val_loss:.4f} at step {best_step}"
    )
    print(
        f"Mean data wait    : "
        f"{summary['mean_data_wait_seconds_per_step']:.3f} s/step"
    )
    print(
        f"Mean GPU compute  : "
        f"{summary['mean_compute_seconds_per_step']:.3f} s/step"
    )
    print(
        f"End-to-end rate   : "
        f"{summary['logical_samples_per_second_end_to_end']:.2f} "
        f"logical samples/s"
    )
    print(f"Summary           : {summary_path}")
    print("=" * 86)


if __name__ == "__main__":
    main()

