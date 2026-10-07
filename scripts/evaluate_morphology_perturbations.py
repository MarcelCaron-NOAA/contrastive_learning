#!/usr/bin/env python3
"""
Expanded controlled morphology perturbation suite.

The experiment uses held-out HRRR validation patches from wang_transform_v1 and
compares frozen learned similarity (z and h) against conventional verification
metrics under explicitly defined shape/organization perturbations.

These probes are diagnostic, not an assertion that any one transform is a
perfect model of real forecast error.  Their job is to reveal which spatial
properties the learned representation is sensitive or invariant to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.verification_metrics import continuous_and_wang_metrics
from src.eval.morphology_transforms import MORPHOLOGY_LEVELS, variants


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/wang_transform_v1",
        "checkpoint": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "outdir": data / "derived/evaluation/representation_characterization_v1/morphology_perturbations",
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def encode_both(model, raw, ptransform, device, batch_size):
    hs, zs = [], []
    for s in range(0, len(raw), batch_size):
        x = torch.stack([ptransform(v) for v in raw[s:s+batch_size]], dim=0)
        x = x.to(device, non_blocking=True)
        h, z = model(x)
        hs.append(F.normalize(h.float(), dim=-1).cpu().numpy())
        zs.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(hs), np.concatenate(zs)


def ci95(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m, 1
    se = float(np.std(x, ddof=1)/np.sqrt(len(x)))
    return m, m-1.96*se, m+1.96*se, len(x)


def total_change_pct(orig, pert):
    so = float(np.sum(orig, dtype=np.float64))
    sp = float(np.sum(pert, dtype=np.float64))
    if so == 0:
        return np.nan
    return 100.0*(sp-so)/so


def plot_family(summary, family, outdir):
    import matplotlib.pyplot as plt
    g = summary[summary.family == family].sort_values("severity")
    plt.figure(figsize=(8.7, 5.8))
    metrics = [
        ("learned_similarity_z_mean", "learned z"),
        ("fss5_mean", "FSS-5"),
        ("fss10_mean", "FSS-10"),
        ("ts_mean", "TS"),
        ("corr_mean", "correlation"),
    ]
    for col, label in metrics:
        if col in g:
            plt.plot(g.severity, g[col], marker="o", label=label)
    plt.xlabel(family)
    plt.ylabel("mean score")
    plt.title(f"Controlled morphology response: {family}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/f"{family}_score_response.png", dpi=190)
    plt.close()


def main():
    d = defaults()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--checkpoint", default=str(d["checkpoint"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--output-dir", default=str(d["outdir"]))
    ap.add_argument("--case-count", type=int, default=500)
    ap.add_argument("--case-seed", type=int, default=20261007)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold-mm", type=float, default=2.54,
                    help="Threshold used for TS and FSS in the conventional metrics.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    archive = Path(args.archive_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_raw = np.load(archive/"original_patches.npz")["qpf6_mm"]
    cases = pd.read_csv(archive/"cases.csv")
    if len(all_raw) != len(cases):
        raise RuntimeError("Archive/cases length mismatch")

    rng = np.random.default_rng(args.case_seed)
    n = min(args.case_count, len(all_raw))
    idx = np.sort(rng.choice(len(all_raw), size=n, replace=False))
    raw = np.asarray(all_raw[idx], dtype=np.float32)
    cases.iloc[idx].to_csv(outdir/"selected_cases.csv", index=False)

    device = torch.device("cuda")
    model, ckpt = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()
    ptransform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    h0, z0 = encode_both(model, raw, ptransform, device, args.batch_size)

    records = []
    families = list(MORPHOLOGY_LEVELS.keys())

    for family in families:
        levels = MORPHOLOGY_LEVELS[family]
        print(f"\n{family}: {len(levels)} severity levels x {n} cases")
        for il, severity in enumerate(levels):
            # Variant count may be 1 or 2. Build each variant name across all cases
            # separately so inference stays batched.
            per_case_variants = [
                variants(raw[i], family, severity, case_seed=args.case_seed + int(idx[i]))
                for i in range(n)
            ]
            variant_names = sorted({name for vv in per_case_variants for name, _ in vv})

            for variant_name in variant_names:
                transformed = []
                case_positions = []
                for i, vv in enumerate(per_case_variants):
                    match = [arr for name, arr in vv if name == variant_name]
                    if match:
                        transformed.append(match[0])
                        case_positions.append(i)
                if not transformed:
                    continue
                transformed = np.stack(transformed).astype(np.float32)
                h1, z1 = encode_both(model, transformed, ptransform, device, args.batch_size)
                pos = np.asarray(case_positions, dtype=int)
                hsim = np.sum(h0[pos]*h1, axis=1)
                zsim = np.sum(z0[pos]*z1, axis=1)

                for j, cp in enumerate(pos):
                    orig = raw[cp]
                    pert = transformed[j]
                    tm = continuous_and_wang_metrics(pert, orig, args.threshold_mm)
                    records.append({
                        "archive_case_index": int(idx[cp]),
                        "case_position": int(cp),
                        "family": family,
                        "severity": float(severity),
                        "variant": variant_name,
                        "learned_similarity_z": float(zsim[j]),
                        "learned_similarity_h": float(hsim[j]),
                        "total_precip_change_pct": total_change_pct(orig, pert),
                        "orig_mean_mm": float(np.mean(orig)),
                        "pert_mean_mm": float(np.mean(pert)),
                        "orig_max_mm": float(np.max(orig)),
                        "pert_max_mm": float(np.max(pert)),
                        **tm,
                    })
            print(f"  {il+1:2d}/{len(levels)} severity={severity:g}")

    per_case = pd.DataFrame(records)
    per_case.to_csv(outdir/"morphology_per_case_metrics.csv.gz", index=False)

    metric_cols = [
        "learned_similarity_z", "learned_similarity_h", "ts", "pod",
        "normalized_far", "fss5", "fss10", "corr", "rmse", "mae",
        "total_precip_change_pct",
    ]
    summary_rows = []
    for (family, severity), g in per_case.groupby(["family","severity"], sort=False):
        row = {"family": family, "severity": severity, "n_rows": len(g)}
        for col in metric_cols:
            mean, lo, hi, nn = ci95(g[col].to_numpy())
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
            row[f"{col}_n"] = nn
        row["abs_total_precip_change_pct_mean"] = float(np.nanmean(np.abs(g.total_precip_change_pct)))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir/"morphology_summary.csv", index=False)

    # A compact ranking of which controlled dimension moves z the most at its
    # maximum tested severity.  This is descriptive, not a universal ordering.
    sensitivity = []
    for family in families:
        g = summary[summary.family == family].sort_values("severity")
        if g.empty:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        sensitivity.append({
            "family": family,
            "baseline_severity": float(first.severity),
            "max_tested_severity": float(last.severity),
            "z_mean_at_baseline": float(first.learned_similarity_z_mean),
            "z_mean_at_max_severity": float(last.learned_similarity_z_mean),
            "z_drop": float(first.learned_similarity_z_mean-last.learned_similarity_z_mean),
            "fss10_drop": float(first.fss10_mean-last.fss10_mean),
            "corr_drop": float(first.corr_mean-last.corr_mean),
        })
    pd.DataFrame(sensitivity).sort_values("z_drop", ascending=False).to_csv(
        outdir/"morphology_sensitivity_ranking.csv", index=False
    )

    for family in families:
        plot_family(summary, family, outdir)

    meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "case_count": n,
        "case_seed": args.case_seed,
        "threshold_mm": args.threshold_mm,
        "families": MORPHOLOGY_LEVELS,
        "archive": str(args.archive_dir),
        "sealed_test_used": False,
        "note": (
            "These are controlled diagnostic probes. Sensitivity indicates what the "
            "frozen representation responds to; it does not by itself establish that "
            "the transformation is a realistic forecast-error model or that larger "
            "sensitivity is always preferable."
        ),
    }
    (outdir/"run_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"\nOutput: {outdir}")


if __name__ == "__main__":
    main()

