#!/usr/bin/env python3
"""
First minimal model/loss test for the HRRR contrastive-learning project.

This is NOT a production training run.

Question:
    Can a plain 1-channel ResNet-18 + nonlinear projection head optimize the
    explicit A/P/N contrastive objective on a few real HRRR minibatches?

Procedure:
    1. Build the finalized training sampler (climo45 by default).
    2. Materialize a small number of real minibatches and cache them in RAM.
       This intentionally separates optimization sanity from filesystem speed.
    3. Measure loss/similarity metrics before training.
    4. Repeatedly cycle over those cached minibatches for a small number of
       optimization steps.
    5. Re-evaluate the exact same cached examples.
    6. Save history, summary JSON, and a checkpoint.

A successful test should show:
    - finite forward/backward passes;
    - parameters actually change;
    - lower symmetric InfoNCE loss;
    - larger A-P versus A-N/P-N cosine margins;
    - higher positive top-1 retrieval than the untrained network.

It does NOT establish generalization.  That comes later with validation data.
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
from src.models.precip_resnet import PrecipResNet18
from src.training.contrastive_loss import SymmetricExplicitInfoNCE


def default_paths():
    data = Path(os.environ["DATA_DIR"])
    exp = data / "derived/experiments/minimal_model_loss_v1"
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": (
            data
            / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": exp,
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        "anchor": batch["anchor"].to(device, non_blocking=True),
        "positive": batch["positive"].to(device, non_blocking=True),
        "negatives": batch["negatives"].to(device, non_blocking=True),
        "negative_mask": batch["negative_mask"].to(device, non_blocking=True),
    }


def encode_batch(model, batch):
    """
    Encode A, P, and explicit N in one model call.

    For default B=4, K=8 this is 40 precipitation patches, which is modest
    on an H100 and gives BatchNorm a reasonable effective batch size.
    """
    a = batch["anchor"]
    p = batch["positive"]
    n = batch["negatives"]

    B, K = n.shape[:2]

    x = torch.cat(
        [
            a,
            p,
            n.reshape(B * K, *n.shape[2:]),
        ],
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


def aggregate_metric_dicts(metric_dicts):
    keys = metric_dicts[0].keys()
    out = {}
    for key in keys:
        vals = []
        for d in metric_dicts:
            v = d[key]
            if torch.is_tensor(v):
                v = float(v.detach().cpu())
            vals.append(float(v))
        out[key] = float(np.mean(vals))
    return out


@torch.no_grad()
def evaluate_cached(model, criterion, cached_batches, device, use_amp):
    model.eval()
    metrics = []

    for cpu_batch in cached_batches:
        batch = move_batch(cpu_batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(use_amp and device.type == "cuda"),
        ):
            _, (z_a, z_p, z_n) = encode_batch(model, batch)
            loss, m = criterion(
                z_a,
                z_p,
                z_n,
                batch["negative_mask"],
            )

        metrics.append(m)

    return aggregate_metric_dicts(metrics)


def parameter_vector_sample(model, max_values=10000):
    vals = []
    remaining = max_values
    with torch.no_grad():
        for p in model.parameters():
            if remaining <= 0:
                break
            flat = p.detach().float().cpu().view(-1)
            take = min(remaining, flat.numel())
            vals.append(flat[:take])
            remaining -= take
    return torch.cat(vals) if vals else torch.empty(0)


def print_metrics(label, m):
    print(
        f"{label:<12s} "
        f"loss={m['loss']:.4f} | "
        f"cos(A,P)={m['cosine_ap']:.4f} | "
        f"cos(A,N)={m['cosine_an']:.4f} | "
        f"marginA={m['margin_a']:+.4f} | "
        f"marginP={m['margin_p']:+.4f} | "
        f"top1A={m['top1_a_pct']:.1f}% | "
        f"top1P={m['top1_p_pct']:.1f}%"
    )


def main():
    defaults = default_paths()

    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(defaults["anchors"]))
    p.add_argument("--negative-pool", default=str(defaults["negative_pool"]))
    p.add_argument("--manifest", default=str(defaults["manifest"]))
    p.add_argument("--stats-json", default=str(defaults["stats"]))
    p.add_argument("--output-dir", default=str(defaults["output_dir"]))

    p.add_argument("--negative-mode", choices=["climo45", "calendar_any"],
                   default="climo45")
    p.add_argument("--negative-count", type=int, default=8)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--cache-batches", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.10)

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--print-every", type=int, default=10)

    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")

    args = p.parse_args()

    if args.batch_size < 2:
        raise SystemExit("--batch-size should be >= 2 for this test")
    if args.cache_batches < 1:
        raise SystemExit("--cache-batches must be >= 1")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise SystemExit(
            "CUDA GPU not available. Run this on an Ursa GPU node "
            "(or pass --allow-cpu only for a very slow debugging run)."
        )

    use_amp = (not args.no_amp) and device.type == "cuda"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MINIMAL HRRR CONTRASTIVE MODEL / LOSS TEST")
    print("=" * 80)
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA capability    : "
            f"{torch.cuda.get_device_capability(0)}"
        )
    print(f"AMP / dtype        : {'bf16' if use_amp else 'off / fp32'}")
    print(f"Negative mode      : {args.negative_mode}")
    print(f"Negatives / sample : {args.negative_count}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Cache batches      : {args.cache_batches}")
    print(f"Training steps     : {args.steps}")
    print(f"Temperature        : {args.temperature}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Weight decay       : {args.weight_decay}")
    print(f"Output             : {outdir}")
    print()

    # ------------------------------------------------------------------
    # Real data pipeline
    # ------------------------------------------------------------------
    index = ContrastiveIndex(
        args.anchors,
        args.negative_pool,
        args.manifest,
    )
    rules = NegativeRules(
        mode=args.negative_mode,
        min_days=args.negative_min_days,
        climo_window_days=args.climo_window_days,
    )
    sampler = ContrastiveBatchSampler(
        index,
        split="train",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=rules,
        seed=args.seed,
    )
    sampler.set_epoch(0)

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore",
        args.stats_json,
    )
    archive = HRRRArchive(args.manifest)
    dataset = ContrastiveDataset(
        archive,
        negative_count=args.negative_count,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=contrastive_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ------------------------------------------------------------------
    # Cache a few REAL minibatches.
    # ------------------------------------------------------------------
    print("-" * 80)
    print("Caching real HRRR minibatches...")
    print("-" * 80)

    cached = []
    t_cache = time.perf_counter()

    try:
        for i, batch in enumerate(
            itertools.islice(loader, args.cache_batches),
            start=1,
        ):
            # Keep cached copies on CPU. DataLoader workers may use pinned
            # memory; clone to simple independent tensors.
            keep = {
                "anchor": batch["anchor"].clone(),
                "positive": batch["positive"].clone(),
                "negatives": batch["negatives"].clone(),
                "negative_mask": batch["negative_mask"].clone(),
            }
            cached.append(keep)

            masked = int((~keep["negative_mask"]).sum())
            print(
                f"Cached batch {i}/{args.cache_batches}: "
                f"A={tuple(keep['anchor'].shape)}, "
                f"N={tuple(keep['negatives'].shape)}, "
                f"masked N={masked}"
            )
    finally:
        archive.close()

    cache_seconds = time.perf_counter() - t_cache

    if len(cached) != args.cache_batches:
        raise RuntimeError(
            f"Expected {args.cache_batches} cached batches, got {len(cached)}"
        )

    print(f"Cache time         : {cache_seconds:.2f} s")
    print()

    # ------------------------------------------------------------------
    # Model and criterion
    # ------------------------------------------------------------------
    model = PrecipResNet18(
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

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("-" * 80)
    print("Model")
    print("-" * 80)
    print("Backbone           : standard 1-channel ResNet-18")
    print(
        f"Projection head    : 512 -> "
        f"{args.projection_hidden_dim} -> {args.projection_dim}"
    )
    print(f"Parameters         : {n_params:,}")
    print(f"Trainable          : {n_trainable:,}")
    print(
        f"Chance top-1       : "
        f"{100.0/(args.negative_count+1):.2f}% "
        f"for 1 positive + {args.negative_count} negatives"
    )
    print()

    # Record a deterministic subset of initial parameters to prove update.
    before_params = parameter_vector_sample(model)

    initial = evaluate_cached(
        model,
        criterion,
        cached,
        device,
        use_amp,
    )
    print_metrics("Initial", initial)
    print()

    # ------------------------------------------------------------------
    # Tiny overfit / optimization sanity test
    # ------------------------------------------------------------------
    history = []
    train_start = time.perf_counter()

    model.train()
    for step in range(1, args.steps + 1):
        cpu_batch = cached[(step - 1) % len(cached)]
        batch = move_batch(cpu_batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
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

        loss.backward()

        # Fail immediately on non-finite gradients.
        finite_grad = True
        sq_norm = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                finite_grad = False
                break
            sq_norm += float(
                torch.sum(param.grad.detach().float() ** 2).cpu()
            )

        if not finite_grad:
            raise RuntimeError(f"Non-finite gradient at step {step}")

        grad_norm = math.sqrt(sq_norm)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(metrics["loss"].cpu()),
            "cosine_ap": float(metrics["cosine_ap"].cpu()),
            "cosine_an": float(metrics["cosine_an"].cpu()),
            "cosine_pn": float(metrics["cosine_pn"].cpu()),
            "margin_a": float(metrics["margin_a"].cpu()),
            "margin_p": float(metrics["margin_p"].cpu()),
            "top1_a_pct": float(metrics["top1_a_pct"].cpu()),
            "top1_p_pct": float(metrics["top1_p_pct"].cpu()),
            "valid_negative_fraction": float(
                metrics["valid_negative_fraction"].cpu()
            ),
            "grad_norm": grad_norm,
        }
        history.append(row)

        if (
            step == 1
            or step % args.print_every == 0
            or step == args.steps
        ):
            print(
                f"step {step:4d}/{args.steps}: "
                f"loss={row['loss']:.4f} "
                f"AP={row['cosine_ap']:.4f} "
                f"AN={row['cosine_an']:.4f} "
                f"marginA={row['margin_a']:+.4f} "
                f"top1A={row['top1_a_pct']:.1f}% "
                f"grad={row['grad_norm']:.3f}"
            )

    train_seconds = time.perf_counter() - train_start

    final = evaluate_cached(
        model,
        criterion,
        cached,
        device,
        use_amp,
    )
    print()
    print_metrics("Final", final)

    after_params = parameter_vector_sample(model)
    param_delta = float(
        torch.linalg.vector_norm(after_params - before_params)
    )

    loss_improved = final["loss"] < initial["loss"]
    margin_a_improved = final["margin_a"] > initial["margin_a"]
    margin_p_improved = final["margin_p"] > initial["margin_p"]
    params_changed = param_delta > 0.0

    optimization_pass = bool(
        loss_improved
        and margin_a_improved
        and margin_p_improved
        and params_changed
    )

    print()
    print("-" * 80)
    print("Optimization sanity")
    print("-" * 80)
    print(
        f"Loss change        : "
        f"{initial['loss']:.4f} -> {final['loss']:.4f} "
        f"({final['loss']-initial['loss']:+.4f})"
    )
    print(
        f"A margin change    : "
        f"{initial['margin_a']:+.4f} -> {final['margin_a']:+.4f}"
    )
    print(
        f"P margin change    : "
        f"{initial['margin_p']:+.4f} -> {final['margin_p']:+.4f}"
    )
    print(f"Parameter Δ norm   : {param_delta:.6f}")
    print(f"Train time         : {train_seconds:.2f} s")
    print(
        f"Step throughput    : "
        f"{args.steps/train_seconds:.2f} steps/s"
    )
    print(
        f"Result             : "
        f"{'PASS' if optimization_pass else 'NEEDS REVIEW'}"
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    history_path = outdir / "training_history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint_path = outdir / "minimal_model_loss_test.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model": {
                "name": "PrecipResNet18",
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
            "sampling": {
                "negative_mode": args.negative_mode,
                "negative_count": args.negative_count,
                "negative_min_days": args.negative_min_days,
                "climo_window_days": args.climo_window_days,
                "patch_size": args.patch_size,
            },
            "seed": args.seed,
        },
        checkpoint_path,
    )

    summary = {
        "status": "PASS" if optimization_pass else "NEEDS_REVIEW",
        "purpose": "minimal optimization sanity test; not a generalization result",
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "amp_bfloat16": bool(use_amp),
        "batch_size": args.batch_size,
        "negative_count": args.negative_count,
        "cache_batches": args.cache_batches,
        "cached_samples": args.batch_size * args.cache_batches,
        "steps": args.steps,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parameter_count": n_params,
        "initial": initial,
        "final": final,
        "loss_change": final["loss"] - initial["loss"],
        "margin_a_change": final["margin_a"] - initial["margin_a"],
        "margin_p_change": final["margin_p"] - initial["margin_p"],
        "parameter_sample_delta_norm": param_delta,
        "cache_seconds": cache_seconds,
        "train_seconds": train_seconds,
        "steps_per_second": args.steps / train_seconds,
        "optimization_pass": optimization_pass,
        "history_csv": str(history_path),
        "checkpoint": str(checkpoint_path),
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"History            : {history_path}")
    print(f"Checkpoint         : {checkpoint_path}")
    print(f"Summary            : {summary_path}")
    print("=" * 80)

    # Deliberately do not exit nonzero on NEEDS_REVIEW: a scientifically
    # interesting optimization failure should preserve all artifacts/logs.
    if not optimization_pass:
        print(
            "The plumbing stayed finite, but the tiny-cache optimization "
            "did not satisfy all expected improvement checks."
        )


if __name__ == "__main__":
    main()

