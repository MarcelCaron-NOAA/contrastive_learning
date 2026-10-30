#!/usr/bin/env python3
"""
Train the final Natural Forecast Variability (NFV) model from scratch.

NFV is the final name for the experiment historically stored on disk as
``hard_ranking_v3_full_v1``.  The legacy experiment identifier is intentionally
retained in checkpoints so existing final checkpoints remain resume-compatible.

For each anchor A, the sampler supplies:
  A, P/TL1, TL3, TL6, E1, E2, E3, E6, N1, N2, N3, N4

Optimized projection-space objective:
    1.0 * [L_global    / log(K+1)]
  + 1.0 * [L_TL        / log(2)]
  + 1.0 * [L_evolution / log(2)]

where K=4 by default. The global term is symmetric explicit-negative InfoNCE,
and TL/evolution are smooth all-pairs ordered-ranking losses. Exact-identical
raw pairs remain masked from strict ranking constraints by the sampler.

This is a clean from-scratch GroupNorm run. No V1/V2 model weights or optimizer
state are loaded unless --resume-checkpoint is explicitly supplied.

The learning-rate schedule matches the prior full GroupNorm run:
  linear warmup over 5% of the complete hard-ranking epoch
  3e-5 -> 3e-4, then cosine decay without restarts to 3e-6.

The complete epoch length is derived from HardRankingBatchSampler at runtime,
so the run uses every hard-eligible training anchor once and never guesses the
step count from the older A/P/N sampler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import ContrastiveIndex, NegativeRules, HRRRArchive, PrecipTransform
from src.data.hard_ranking_sampler import (
    HardRankingBatchSampler,
    HardRankingDataset,
    hard_ranking_collate,
)
from src.models.precip_resnet_gn import PrecipResNet18GN
from src.training.contrastive_loss import SymmetricExplicitInfoNCE
from src.training.ordered_ranking_loss import OrderedSimilarityRankingLoss


EXPERIMENT_ID = "hard_ranking_v3_full_v1"
MODEL_LABEL = "NFV"


TL_PAIR_LABELS = ("p_gt_tl3", "p_gt_tl6", "tl3_gt_tl6")
E_PAIR_LABELS = (
    "e1_gt_e2", "e1_gt_e3", "e1_gt_e6",
    "e2_gt_e3", "e2_gt_e6",
    "e3_gt_e6",
)


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "anchors": data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz",
        "negative_pool": data / "metadata/dataset_v1/negative_pool_v1.csv.gz",
        "manifest": data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "output_dir": data / "derived/experiments/hard_ranking_v3_full_v1",
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shutdown_iterator(iterator):
    fn = getattr(iterator, "_shutdown_workers", None)
    if callable(fn):
        fn()


def loader_kwargs(dataset, sampler, workers, pin_memory, prefetch_factor):
    kw = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=hard_ranking_collate,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = prefetch_factor
    return kw


def move_batch(batch, device):
    keys = (
        "anchor", "positive", "tl3", "tl6",
        "e1", "e2", "e3", "e6",
        "negatives", "negative_mask", "tl_pair_mask", "evolution_pair_mask",
    )
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in keys
    }


def encode_batch(model, batch, use_amp):
    """One model forward for all 8+K fields; return h/z dictionaries."""
    B = batch["anchor"].shape[0]
    K = batch["negatives"].shape[1]
    ordered = [
        batch["anchor"], batch["positive"], batch["tl3"], batch["tl6"],
        batch["e1"], batch["e2"], batch["e3"], batch["e6"],
        batch["negatives"].reshape(B * K, *batch["negatives"].shape[2:]),
    ]
    x = torch.cat(ordered, dim=0)

    with torch.autocast(
        device_type=x.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        h, z = model(x)

    # Ranking differences can be subtle near cosine 1. Use FP32 for all
    # similarity/loss computations even when the network forward uses bf16.
    h = h.float()
    z = z.float()

    cuts = [B * i for i in range(9)]
    h_parts = [h[cuts[i]:cuts[i + 1]] for i in range(8)]
    z_parts = [z[cuts[i]:cuts[i + 1]] for i in range(8)]
    h_n = h[cuts[8]:].reshape(B, K, -1)
    z_n = z[cuts[8]:].reshape(B, K, -1)

    names = ("a", "p", "tl3", "tl6", "e1", "e2", "e3", "e6")
    h_out = {name: value for name, value in zip(names, h_parts)}
    z_out = {name: value for name, value in zip(names, z_parts)}
    h_out["n"] = h_n
    z_out["n"] = z_n
    return h_out, z_out


def make_rank_views(rep):
    tl = torch.stack([rep["p"], rep["tl3"], rep["tl6"]], dim=1)
    evo = torch.stack([rep["e1"], rep["e2"], rep["e3"], rep["e6"]], dim=1)
    return tl, evo


def compute_z_objective(
    z,
    batch,
    global_criterion,
    rank_criterion,
    *,
    lambda_global,
    lambda_tl,
    lambda_evolution,
    negative_count,
):
    global_raw, global_metrics = global_criterion(
        z["a"], z["p"], z["n"], batch["negative_mask"]
    )
    tl_views, evo_views = make_rank_views(z)
    tl_raw, tl_metrics = rank_criterion(
        z["a"], tl_views, batch["tl_pair_mask"]
    )
    evo_raw, evo_metrics = rank_criterion(
        z["a"], evo_views, batch["evolution_pair_mask"]
    )

    global_norm = global_raw / math.log(negative_count + 1.0)
    tl_norm = tl_raw / math.log(2.0)
    evo_norm = evo_raw / math.log(2.0)

    total = (
        lambda_global * global_norm
        + lambda_tl * tl_norm
        + lambda_evolution * evo_norm
    )

    return total, {
        "global_raw": global_raw,
        "global_norm": global_norm,
        "global_metrics": global_metrics,
        "tl_raw": tl_raw,
        "tl_norm": tl_norm,
        "tl_metrics": tl_metrics,
        "evolution_raw": evo_raw,
        "evolution_norm": evo_norm,
        "evolution_metrics": evo_metrics,
        "global_contribution": lambda_global * global_norm,
        "tl_contribution": lambda_tl * tl_norm,
        "evolution_contribution": lambda_evolution * evo_norm,
    }


def pair_stats(anchor, views, pair_mask):
    """Exact aggregation primitives for validation reporting."""
    sims = torch.einsum("bd,bmd->bm", anchor, views)
    pairs = [(i, j) for i in range(sims.shape[1]) for j in range(i + 1, sims.shape[1])]
    good = torch.stack([sims[:, i] > sims[:, j] for i, j in pairs], dim=1)
    mask = pair_mask.bool()

    pair_good = (good & mask).sum(dim=0).detach().cpu().numpy().astype(np.int64)
    pair_valid = mask.sum(dim=0).detach().cpu().numpy().astype(np.int64)

    adjacent_idx = [q for q, (i, j) in enumerate(pairs) if j == i + 1]
    adj_mask = mask[:, adjacent_idx]
    eligible = adj_mask.all(dim=1)
    full_good = good[:, adjacent_idx].all(dim=1) & eligible

    return {
        "sim_sum": sims.sum(dim=0).detach().cpu().numpy().astype(np.float64),
        "n_samples": int(sims.shape[0]),
        "pair_good": pair_good,
        "pair_valid": pair_valid,
        "full_good": int(full_good.sum().item()),
        "full_valid": int(eligible.sum().item()),
    }


@torch.no_grad()
def evaluate_cached(
    model,
    cached_validation,
    device,
    use_amp,
    global_criterion,
    rank_criterion,
    args,
):
    model.eval()

    out = {}
    for space in ("z", "h"):
        out[space] = {
            "global_loss_sum": 0.0,
            "global_weight": 0,
            "global_ap_sum": 0.0,
            "global_an_sum": 0.0,
            "global_an_count": 0,
            "global_margin_sum": 0.0,
            "global_top1_sum": 0.0,
            "tl_loss_sum": 0.0,
            "tl_valid_pairs": 0,
            "evo_loss_sum": 0.0,
            "evo_valid_pairs": 0,
            "tl": None,
            "evo": None,
        }

    for cpu_batch in cached_validation:
        batch = move_batch(cpu_batch, device)
        h, z = encode_batch(model, batch, use_amp)
        reps = {"z": z, "h": {k: F.normalize(v, dim=-1) for k, v in h.items()}}

        for space, rep in reps.items():
            acc = out[space]
            global_raw, gm = global_criterion(
                rep["a"], rep["p"], rep["n"], batch["negative_mask"]
            )
            active = int(gm["active_query_count"].item())
            if active > 0:
                acc["global_loss_sum"] += float(global_raw.item()) * active
                acc["global_weight"] += active
                acc["global_ap_sum"] += float(gm["cosine_ap"].item()) * active
                acc["global_margin_sum"] += float(gm["margin_a"].item()) * active
                acc["global_top1_sum"] += float(gm["top1_a_pct"].item()) * active

            # Mean AN is over valid negative slots, not active logical samples.
            valid_n = int(batch["negative_mask"].sum().item())
            if valid_n > 0:
                acc["global_an_sum"] += float(gm["cosine_an"].item()) * valid_n
                acc["global_an_count"] += valid_n

            tl_views, evo_views = make_rank_views(rep)
            tl_raw, tlm = rank_criterion(rep["a"], tl_views, batch["tl_pair_mask"])
            evo_raw, em = rank_criterion(rep["a"], evo_views, batch["evolution_pair_mask"])

            tl_pairs = int(tlm["valid_pair_count"].item())
            evo_pairs = int(em["valid_pair_count"].item())
            if tl_pairs > 0:
                acc["tl_loss_sum"] += float(tl_raw.item()) * tl_pairs
                acc["tl_valid_pairs"] += tl_pairs
            if evo_pairs > 0:
                acc["evo_loss_sum"] += float(evo_raw.item()) * evo_pairs
                acc["evo_valid_pairs"] += evo_pairs

            tstat = pair_stats(rep["a"], tl_views, batch["tl_pair_mask"])
            estat = pair_stats(rep["a"], evo_views, batch["evolution_pair_mask"])

            for key, stat in (("tl", tstat), ("evo", estat)):
                if acc[key] is None:
                    acc[key] = stat
                else:
                    acc[key]["sim_sum"] += stat["sim_sum"]
                    acc[key]["n_samples"] += stat["n_samples"]
                    acc[key]["pair_good"] += stat["pair_good"]
                    acc[key]["pair_valid"] += stat["pair_valid"]
                    acc[key]["full_good"] += stat["full_good"]
                    acc[key]["full_valid"] += stat["full_valid"]

    result = {}
    for space, acc in out.items():
        prefix = f"{space}_"
        gw = max(1, acc["global_weight"])
        global_raw = acc["global_loss_sum"] / gw
        tl_raw = acc["tl_loss_sum"] / max(1, acc["tl_valid_pairs"])
        evo_raw = acc["evo_loss_sum"] / max(1, acc["evo_valid_pairs"])
        global_norm = global_raw / math.log(args.negative_count + 1.0)
        tl_norm = tl_raw / math.log(2.0)
        evo_norm = evo_raw / math.log(2.0)

        result[prefix + "global_raw_loss"] = global_raw
        result[prefix + "global_norm_loss"] = global_norm
        result[prefix + "global_ap"] = acc["global_ap_sum"] / gw
        result[prefix + "global_an"] = acc["global_an_sum"] / max(1, acc["global_an_count"])
        result[prefix + "global_margin_a"] = acc["global_margin_sum"] / gw
        result[prefix + "global_top1_a_pct"] = acc["global_top1_sum"] / gw
        result[prefix + "tl_raw_loss"] = tl_raw
        result[prefix + "tl_norm_loss"] = tl_norm
        result[prefix + "evolution_raw_loss"] = evo_raw
        result[prefix + "evolution_norm_loss"] = evo_norm

        for label, value in zip(
            ("p", "tl3", "tl6"),
            acc["tl"]["sim_sum"] / acc["tl"]["n_samples"],
        ):
            result[prefix + f"sim_{label}"] = float(value)
        for label, value in zip(
            ("e1", "e2", "e3", "e6"),
            acc["evo"]["sim_sum"] / acc["evo"]["n_samples"],
        ):
            result[prefix + f"sim_{label}"] = float(value)

        for label, good, valid in zip(
            TL_PAIR_LABELS, acc["tl"]["pair_good"], acc["tl"]["pair_valid"]
        ):
            result[prefix + label + "_pct"] = (100.0 * good / valid) if valid else float("nan")
        result[prefix + "tl_full_order_pct"] = (
            100.0 * acc["tl"]["full_good"] / acc["tl"]["full_valid"]
            if acc["tl"]["full_valid"] else float("nan")
        )

        for label, good, valid in zip(
            E_PAIR_LABELS, acc["evo"]["pair_good"], acc["evo"]["pair_valid"]
        ):
            result[prefix + label + "_pct"] = (100.0 * good / valid) if valid else float("nan")
        result[prefix + "evolution_full_order_pct"] = (
            100.0 * acc["evo"]["full_good"] / acc["evo"]["full_valid"]
            if acc["evo"]["full_valid"] else float("nan")
        )

        result[prefix + "total_objective"] = (
            args.lambda_global * global_norm
            + args.lambda_tl * tl_norm
            + args.lambda_evolution * evo_norm
        )

    return result


def learning_rate_for_step(step, total_steps, warmup_steps, start_lr, peak_lr, min_lr):
    step = int(step)
    if warmup_steps > 0 and step <= warmup_steps:
        if warmup_steps == 1:
            return float(peak_lr)
        frac = (step - 1) / float(warmup_steps - 1)
        return float(start_lr + frac * (peak_lr - start_lr))

    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / float(denom)
    progress = min(1.0, max(0.0, progress))
    return float(
        min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    )


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_checkpoint(
    path, *, model, optimizer, step, best_val, args,
    full_epoch_steps, warmup_steps, hard_eligible_anchors,
):
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "experiment": EXPERIMENT_ID,
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_z_total_objective": float(best_val["loss"]),
            "best_val_step": int(best_val["step"]),
            "model": {
                "name": "PrecipResNet18GN",
                "normalization": "GroupNorm32",
                "projection_hidden_dim": args.projection_hidden_dim,
                "projection_dim": args.projection_dim,
            },
            "objective": {
                "global": "SymmetricExplicitInfoNCE",
                "global_temperature": args.global_temperature,
                "ranking": "all-pairs softplus order loss",
                "ranking_temperature": args.ranking_temperature,
                "lambda_global": args.lambda_global,
                "lambda_tl": args.lambda_tl,
                "lambda_evolution": args.lambda_evolution,
                "global_normalizer": f"log({args.negative_count + 1})",
                "ranking_normalizer": "log(2)",
            },
            "sampling": {
                "negative_count": args.negative_count,
                "training_negative_mode": "climo45",
                "validation_negative_mode": "calendar_any",
                "negative_min_days": args.negative_min_days,
                "climo_window_days": args.climo_window_days,
                "patch_size": args.patch_size,
                "hard_eligible_anchors": int(hard_eligible_anchors),
                "full_epoch_steps": int(full_epoch_steps),
            },
            "learning_rate_schedule": {
                "type": "linear_warmup_then_cosine_decay_no_restarts",
                "warmup_start_lr": args.warmup_start_lr,
                "peak_lr": args.learning_rate,
                "min_lr": args.min_learning_rate,
                "warmup_fraction": args.warmup_fraction,
                "warmup_steps": int(warmup_steps),
                "full_epoch_steps": int(full_epoch_steps),
            },
            "seed": args.seed,
            "validation_seed": args.validation_seed,
        },
        path,
    )


def validate_recipe_contract(index, recipes):
    """Mechanically verify the scientific relation for sampled recipes."""
    for r in recipes:
        a = index.field_metadata(r.anchor_id)
        a_valid = int(a["valid_unix_s"])
        a_fhr = int(a["forecast_hour"])
        a_init = a_valid - a_fhr * 3600
        s = int(r.branch_sign)

        expected_same_valid = {
            r.positive_id: a_fhr + s,
            r.tl3_id: a_fhr + 3 * s,
            r.tl6_id: a_fhr + 6 * s,
        }
        for sid, efhr in expected_same_valid.items():
            m = index.field_metadata(sid)
            assert int(m["valid_unix_s"]) == a_valid
            assert int(m["forecast_hour"]) == efhr

        expected_evo = {
            r.e1_id: 1,
            r.e2_id: 2,
            r.e3_id: 3,
            r.e6_id: 6,
        }
        for sid, off in expected_evo.items():
            m = index.field_metadata(sid)
            valid = int(m["valid_unix_s"])
            fhr = int(m["forecast_hour"])
            assert valid == a_valid + s * off * 3600
            assert fhr == a_fhr + s * off
            assert valid - fhr * 3600 == a_init

        # Global negatives retain same lead/hour and > configured time distance;
        # detailed climo45/calendar_any checks remain in ContrastiveIndex sampler.
        for nid in r.negative_ids:
            n = index.field_metadata(nid)
            assert int(n["forecast_hour"]) == a_fhr
            assert int(n["valid_hour_utc"]) == int(a["valid_hour_utc"])


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default=str(d["anchors"]))
    p.add_argument("--negative-pool", default=str(d["negative_pool"]))
    p.add_argument("--manifest", default=str(d["manifest"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--resume-checkpoint", default=None)
    p.add_argument("--output-dir", default=str(d["output_dir"]))

    p.add_argument(
        "--end-step", type=int, default=None,
        help="Optional absolute stopping step for debugging/recovery. Default: full hard-ranking epoch.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negative-count", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--negative-min-days", type=float, default=90.0)
    p.add_argument("--climo-window-days", type=int, default=45)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)

    p.add_argument("--validation-samples", type=int, default=512)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=2500)

    p.add_argument("--projection-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--global-temperature", type=float, default=0.10)
    p.add_argument("--ranking-temperature", type=float, default=0.10)
    p.add_argument("--lambda-global", type=float, default=1.0)
    p.add_argument("--lambda-tl", type=float, default=1.0)
    p.add_argument("--lambda-evolution", type=float, default=1.0)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-start-lr", type=float, default=3e-5)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--min-learning-rate", type=float, default=3e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--validation-seed", type=int, default=20261006)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    if args.negative_count < 1:
        raise SystemExit("--negative-count must be >=1")
    if args.validation_samples % args.batch_size != 0:
        raise SystemExit("--validation-samples must be divisible by batch size")
    if not (0.0 <= args.warmup_fraction < 1.0):
        raise SystemExit("--warmup-fraction must be in [0,1)")
    if args.end_step is not None and args.end_step < 1:
        raise SystemExit("--end-step must be >=1 when supplied")
    if not (0 < args.min_learning_rate <= args.learning_rate):
        raise SystemExit("min LR must be >0 and <= peak LR")
    if not (0 < args.warmup_start_lr <= args.learning_rate):
        raise SystemExit("warmup start LR must be >0 and <= peak LR")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    use_amp = not args.no_amp

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_path = outdir / "config.json"
    if config_path.exists() and args.resume_checkpoint is None:
        raise RuntimeError(
            f"{config_path} already exists. Refusing to overwrite an existing run. "
            "Use a new --output-dir or --resume-checkpoint."
        )

    print("=" * 96)
    print("NFV / NATURAL FORECAST VARIABILITY TRAINING")
    print("=" * 96)
    print(f"GPU                     : {torch.cuda.get_device_name(0)}")
    print(f"AMP forward             : {'bf16' if use_amp else 'off'}")
    print("Loss/similarity math    : FP32")
    print("Initialization          : fresh random GroupNorm weights")
    print(f"Output                  : {outdir}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Unrelated negatives K   : {args.negative_count}")
    print(f"Physical fields / step  : {args.batch_size * (8 + args.negative_count)}")
    print(f"Lambdas                 : global={args.lambda_global} TL={args.lambda_tl} evolution={args.lambda_evolution}")
    print(f"Peak LR                 : {args.learning_rate:.3e}")
    print(f"Warmup fraction         : {args.warmup_fraction:.3f} from {args.warmup_start_lr:.3e}")
    print(f"Minimum LR              : {args.min_learning_rate:.3e}")
    print()

    print("Loading metadata index...")
    index = ContrastiveIndex(args.anchors, args.negative_pool, args.manifest)
    transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)

    train_sampler = HardRankingBatchSampler(
        index,
        split="train",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=NegativeRules(
            mode="climo45",
            min_days=args.negative_min_days,
            climo_window_days=args.climo_window_days,
        ),
        seed=args.seed,
    )
    train_sampler.set_epoch(0)

    val_sampler = HardRankingBatchSampler(
        index,
        split="validation",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        negative_count=args.negative_count,
        negative_rules=NegativeRules(
            mode="calendar_any",
            min_days=args.negative_min_days,
            climo_window_days=args.climo_window_days,
        ),
        seed=args.validation_seed,
    )
    val_sampler.set_epoch(0)


    print(f"Hard-eligible train anchors : {train_sampler.n_anchors:,} ({len(train_sampler):,} batches)")
    print(f"Hard-eligible val anchors   : {val_sampler.n_anchors:,} ({len(val_sampler):,} batches)")
    print(f"Train branch availability   : {train_sampler.branch_counts}")
    print(f"Val branch availability     : {val_sampler.branch_counts}")

    full_epoch_steps = len(train_sampler)
    hard_eligible_anchors = int(train_sampler.n_anchors)
    target_end_step = full_epoch_steps if args.end_step is None else int(args.end_step)
    if target_end_step > full_epoch_steps:
        raise RuntimeError(
            f"--end-step={target_end_step} exceeds the complete deterministic "
            f"hard-ranking epoch of {full_epoch_steps} batches."
        )
    warmup_steps = max(
        1, int(round(full_epoch_steps * args.warmup_fraction))
    )
    if warmup_steps >= full_epoch_steps:
        raise RuntimeError("Warmup consumes the complete epoch; reduce --warmup-fraction.")

    print(f"Full NFV anchor epoch       : {hard_eligible_anchors:,} anchors in {full_epoch_steps:,} batches")
    print(f"Target end step             : {target_end_step:,}")
    print("LR schedule reference       : complete hard-ranking epoch (never compressed)")
    print(f"Warmup steps                : {warmup_steps:,}")

    # Mechanical contract test on the first 256 recipes before any I/O-heavy work.
    contract_recipes = []
    for batch_recipes in train_sampler:
        contract_recipes.extend(batch_recipes)
        if len(contract_recipes) >= 256:
            break
    validate_recipe_contract(index, contract_recipes[:256])
    print("Hard-ranking recipe contract: PASS (256 sampled training recipes)")
    print()

    # ------------------------------------------------------------------
    # Fixed validation cache
    # ------------------------------------------------------------------
    print("Building fixed hard-ranking validation cache...")
    val_archive = HRRRArchive(args.manifest)
    val_dataset = HardRankingDataset(
        val_archive,
        negative_count=args.negative_count,
        transform=transform,
    )
    val_loader = DataLoader(
        **loader_kwargs(
            val_dataset, val_sampler, args.num_workers, True, args.prefetch_factor
        )
    )
    val_iter = iter(val_loader)
    n_val_batches = args.validation_samples // args.batch_size
    cached_validation = []
    validation_recipes = []
    t0 = time.perf_counter()
    try:
        for i in range(n_val_batches):
            b = next(val_iter)
            cached_validation.append({
                key: b[key].clone()
                for key in (
                    "anchor", "positive", "tl3", "tl6",
                    "e1", "e2", "e3", "e6", "negatives",
                    "negative_mask", "tl_pair_mask", "evolution_pair_mask",
                )
            })
            validation_recipes.extend(b["metadata"])
            if i == 0 or (i + 1) % 16 == 0 or i + 1 == n_val_batches:
                print(f"  cached {i+1:3d}/{n_val_batches} batches")
    finally:
        shutdown_iterator(val_iter)
        val_archive.close()
        del val_iter, val_loader, val_dataset, val_archive

    recipe_path = outdir / "fixed_validation_recipes.jsonl"
    recipe_text = "".join(json.dumps(x, sort_keys=True) + "\n" for x in validation_recipes)
    if recipe_path.exists():
        if recipe_path.read_text() != recipe_text:
            raise RuntimeError("Fixed validation recipes changed on resume")
    else:
        recipe_path.write_text(recipe_text)
    print(f"Validation cache: {args.validation_samples} samples in {time.perf_counter()-t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Model and optimizer
    # ------------------------------------------------------------------
    # Re-seed immediately before initialization so random weights do not depend
    # on how many RNG draws were consumed while building the validation cache.
    seed_everything(args.seed)
    model = PrecipResNet18GN(
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)

    global_criterion = SymmetricExplicitInfoNCE(args.global_temperature)
    rank_criterion = OrderedSimilarityRankingLoss(args.ranking_temperature)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    best_val = {"loss": float("inf"), "step": 0}

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        ckpt = torch.load(resume_path, map_location="cpu")
        if ckpt.get("experiment") != EXPERIMENT_ID:
            raise RuntimeError(
                f"Resume checkpoint is not compatible with NFV ({EXPERIMENT_ID})"
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        best_val = {
            "loss": float(ckpt["best_val_z_total_objective"]),
            "step": int(ckpt["best_val_step"]),
        }
        if start_step >= target_end_step:
            raise RuntimeError(
                f"Resume checkpoint step={start_step} has already reached "
                f"target_end_step={target_end_step}."
            )
        sampling_meta = ckpt.get("sampling", {})
        if int(sampling_meta.get("full_epoch_steps", full_epoch_steps)) != full_epoch_steps:
            raise RuntimeError("Resume checkpoint full-epoch length differs from current sampler")
        if int(sampling_meta.get("hard_eligible_anchors", hard_eligible_anchors)) != hard_eligible_anchors:
            raise RuntimeError("Resume checkpoint hard-eligible anchor count differs")
        print(f"Resuming NFV checkpoint    : {resume_path} at step {start_step:,}")
    else:
        print("Model weights               : FRESH random initialization")
        print("AdamW optimizer state       : FRESH")

    # Preserve canonical config for GitHub/reproducibility.
    if not config_path.exists():
        config = dict(vars(args))
        config["experiment"] = EXPERIMENT_ID
        config["model_label"] = MODEL_LABEL
        config["initialization"] = "fresh_random_groupnorm"
        config["hard_eligible_anchors"] = hard_eligible_anchors
        config["full_epoch_steps"] = full_epoch_steps
        config["target_end_step"] = target_end_step
        config["warmup_steps"] = warmup_steps
        config["loss_formula"] = (
            "lambda_global*(L_global/log(K+1)) + "
            "lambda_tl*(L_tl/log(2)) + lambda_evolution*(L_evolution/log(2))"
        )
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    # Histories: on resume, keep rows through the checkpoint step only.
    train_hist_path = outdir / "training_history.csv"
    val_hist_path = outdir / "validation_history.csv"
    train_rows = []
    val_rows = []
    if args.resume_checkpoint:
        import pandas as pd
        if train_hist_path.exists():
            train_rows = pd.read_csv(train_hist_path).query("step <= @start_step").to_dict("records")
        if val_hist_path.exists():
            val_rows = pd.read_csv(val_hist_path).query("step <= @start_step").to_dict("records")

    # Baseline validation at random-initialization step 0 is scientifically important.
    if start_step == 0:
        metrics0 = evaluate_cached(
            model, cached_validation, device, use_amp,
            global_criterion, rank_criterion, args,
        )
        row0 = {"step": 0, "learning_rate": 0.0, **metrics0}
        val_rows.append(row0)
        print(
            "VAL step=    0 | "
            f"total={metrics0['z_total_objective']:.4f} | "
            f"global={metrics0['z_global_norm_loss']:.4f} top1={metrics0['z_global_top1_a_pct']:.1f}% | "
            f"TL={metrics0['z_tl_norm_loss']:.4f} full={metrics0['z_tl_full_order_pct']:.1f}% | "
            f"E={metrics0['z_evolution_norm_loss']:.4f} full={metrics0['z_evolution_full_order_pct']:.1f}%"
        )
        best_val = {"loss": float(metrics0["z_total_objective"]), "step": 0}
        save_checkpoint(
            outdir / "checkpoint_best.pt",
            model=model, optimizer=optimizer, step=0,
            best_val=best_val, args=args,
            full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
            hard_eligible_anchors=hard_eligible_anchors,
        )
        write_csv(val_hist_path, val_rows)

    train_archive = HRRRArchive(args.manifest)
    train_dataset = HardRankingDataset(
        train_archive,
        negative_count=args.negative_count,
        transform=transform,
    )

    # Skip completed deterministic recipe batches on resume.
    base_iter = iter(train_sampler)
    recipe_iter = itertools.islice(base_iter, start_step, target_end_step)

    # DataLoader needs a batch_sampler object, so wrap the sliced recipe iterator.
    class _RecipeSlice:
        def __iter__(self_nonlocal):
            return recipe_iter
        def __len__(self_nonlocal):
            return target_end_step - start_step

    train_loader = DataLoader(
        **loader_kwargs(
            train_dataset, _RecipeSlice(), args.num_workers, True, args.prefetch_factor
        )
    )
    train_iter = iter(train_loader)

    data_wait_sum = 0.0
    compute_sum = 0.0
    run_start = time.perf_counter()

    try:
        for step in range(start_step + 1, target_end_step + 1):
            wait0 = time.perf_counter()
            cpu_batch = next(train_iter)
            data_wait = time.perf_counter() - wait0
            data_wait_sum += data_wait

            lr = learning_rate_for_step(
                step, full_epoch_steps, warmup_steps,
                args.warmup_start_lr, args.learning_rate, args.min_learning_rate,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr

            compute0 = time.perf_counter()
            batch = move_batch(cpu_batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            _, z = encode_batch(model, batch, use_amp)
            total, pieces = compute_z_objective(
                z, batch, global_criterion, rank_criterion,
                lambda_global=args.lambda_global,
                lambda_tl=args.lambda_tl,
                lambda_evolution=args.lambda_evolution,
                negative_count=args.negative_count,
            )
            total.backward()
            optimizer.step()
            compute_sec = time.perf_counter() - compute0
            compute_sum += compute_sec

            tlm = pieces["tl_metrics"]
            em = pieces["evolution_metrics"]
            gm = pieces["global_metrics"]
            row = {
                "step": step,
                "learning_rate": lr,
                "total_loss": float(total.detach().cpu()),
                "global_raw_loss": float(pieces["global_raw"].detach().cpu()),
                "global_norm_loss": float(pieces["global_norm"].detach().cpu()),
                "global_contribution": float(pieces["global_contribution"].detach().cpu()),
                "global_top1_a_pct": float(gm["top1_a_pct"].cpu()),
                "global_margin_a": float(gm["margin_a"].cpu()),
                "tl_raw_loss": float(pieces["tl_raw"].detach().cpu()),
                "tl_norm_loss": float(pieces["tl_norm"].detach().cpu()),
                "tl_contribution": float(pieces["tl_contribution"].detach().cpu()),
                "tl_pairwise_order_pct": float(tlm["pairwise_order_pct"].cpu()),
                "tl_full_order_pct": float(tlm["full_order_pct"].cpu()),
                "evolution_raw_loss": float(pieces["evolution_raw"].detach().cpu()),
                "evolution_norm_loss": float(pieces["evolution_norm"].detach().cpu()),
                "evolution_contribution": float(pieces["evolution_contribution"].detach().cpu()),
                "evolution_pairwise_order_pct": float(em["pairwise_order_pct"].cpu()),
                "evolution_full_order_pct": float(em["full_order_pct"].cpu()),
                "data_wait_seconds": data_wait,
                "gpu_compute_seconds": compute_sec,
            }
            train_rows.append(row)

            if step == 1 or step % args.print_every == 0:
                print(
                    f"TRAIN step={step:5d}/{target_end_step} (epoch={step/full_epoch_steps:.3f}) lr={lr:.3e} | "
                    f"total={row['total_loss']:.4f} | "
                    f"G={row['global_norm_loss']:.3f}({row['global_contribution']:.3f}) "
                    f"TL={row['tl_norm_loss']:.3f}({row['tl_contribution']:.3f}) "
                    f"E={row['evolution_norm_loss']:.3f}({row['evolution_contribution']:.3f}) | "
                    f"Gtop1={row['global_top1_a_pct']:.1f}% "
                    f"TLfull={row['tl_full_order_pct']:.1f}% "
                    f"Efull={row['evolution_full_order_pct']:.1f}% | "
                    f"wait={data_wait:.3f}s compute={compute_sec:.3f}s"
                )

            if step % args.validate_every == 0 or step == target_end_step:
                vm = evaluate_cached(
                    model, cached_validation, device, use_amp,
                    global_criterion, rank_criterion, args,
                )
                vrow = {"step": step, "learning_rate": lr, **vm}
                val_rows.append(vrow)
                print(
                    f"VAL step={step:5d} | total={vm['z_total_objective']:.4f} | "
                    f"global={vm['z_global_norm_loss']:.4f} top1={vm['z_global_top1_a_pct']:.1f}% | "
                    f"TL={vm['z_tl_norm_loss']:.4f} full={vm['z_tl_full_order_pct']:.1f}% "
                    f"S={vm['z_sim_p']:.3f}>{vm['z_sim_tl3']:.3f}>{vm['z_sim_tl6']:.3f} | "
                    f"E={vm['z_evolution_norm_loss']:.4f} full={vm['z_evolution_full_order_pct']:.1f}% "
                    f"S={vm['z_sim_e1']:.3f}>{vm['z_sim_e2']:.3f}>{vm['z_sim_e3']:.3f}>{vm['z_sim_e6']:.3f}"
                )

                if vm["z_total_objective"] < best_val["loss"]:
                    best_val = {"loss": float(vm["z_total_objective"]), "step": step}
                    save_checkpoint(
                        outdir / "checkpoint_best.pt",
                        model=model, optimizer=optimizer, step=step,
                        best_val=best_val, args=args,
                        full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
                        hard_eligible_anchors=hard_eligible_anchors,
                    )
                    print(f"  NEW BEST composite validation objective at NFV step {step}")

                write_csv(val_hist_path, val_rows)

            if step % args.checkpoint_every == 0 and step < target_end_step:
                save_checkpoint(
                    outdir / f"checkpoint_step{step:05d}.pt",
                    model=model, optimizer=optimizer, step=step,
                    best_val=best_val, args=args,
                    full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
                    hard_eligible_anchors=hard_eligible_anchors,
                )
                write_csv(train_hist_path, train_rows)

        write_csv(train_hist_path, train_rows)
        write_csv(val_hist_path, val_rows)
        save_checkpoint(
            outdir / "checkpoint_last.pt",
            model=model, optimizer=optimizer, step=target_end_step,
            best_val=best_val, args=args,
            full_epoch_steps=full_epoch_steps, warmup_steps=warmup_steps,
            hard_eligible_anchors=hard_eligible_anchors,
        )

    finally:
        shutdown_iterator(train_iter)
        train_archive.close()

    elapsed = time.perf_counter() - run_start
    summary = {
        "experiment": EXPERIMENT_ID,
        "model_label": MODEL_LABEL,
        "initialization": "fresh_random_groupnorm" if start_step == 0 else "resumed_nfv",
        "hard_eligible_anchors": hard_eligible_anchors,
        "full_epoch_steps": full_epoch_steps,
        "target_end_step": target_end_step,
        "completed_fraction_of_epoch": target_end_step / full_epoch_steps,
        "best_validation": best_val,
        "elapsed_seconds": elapsed,
        "mean_data_wait_seconds_per_step": data_wait_sum / max(1, target_end_step - start_step),
        "mean_gpu_compute_seconds_per_step": compute_sum / max(1, target_end_step - start_step),
        "objective": {
            "lambda_global": args.lambda_global,
            "lambda_tl": args.lambda_tl,
            "lambda_evolution": args.lambda_evolution,
            "global_normalizer": f"log({args.negative_count + 1})",
            "ranking_normalizer": "log(2)",
        },
        "learning_rate_schedule": {
            "type": "linear_warmup_then_cosine_decay_no_restarts",
            "warmup_fraction": args.warmup_fraction,
            "warmup_steps": warmup_steps,
            "warmup_start_lr": args.warmup_start_lr,
            "peak_lr": args.learning_rate,
            "min_lr": args.min_learning_rate,
            "schedule_reference_steps": full_epoch_steps,
        },
        "checkpoint_best": str(outdir / "checkpoint_best.pt"),
        "checkpoint_last": str(outdir / "checkpoint_last.pt"),
        "validation_history": str(val_hist_path),
        "training_history": str(train_hist_path),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("=" * 96)
    print("RUN COMPLETE")
    print("=" * 96)
    print(f"Best validation composite : {best_val['loss']:.6f} at NFV step {best_val['step']}")
    print(f"Mean data wait / step     : {summary['mean_data_wait_seconds_per_step']:.3f}s")
    print(f"Mean GPU compute / step   : {summary['mean_gpu_compute_seconds_per_step']:.3f}s")
    print(f"Output                    : {outdir}")


if __name__ == "__main__":
    main()

