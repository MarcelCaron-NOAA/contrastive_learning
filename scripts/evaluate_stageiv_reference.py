#!/usr/bin/env python3
"""
Evaluate the frozen v1 learned similarity on HRRR-vs-Stage-IV validation pairs.

Primary observation remap: BUDGET
Sensitivity remap: BILIN

Outputs:
  per_case_metrics.csv.gz
  score_summary.csv
  regrid_sensitivity.json
  diagnostic scatter/histogram plots

This script deliberately evaluates the same raw Stage-IV field under two
regridding choices before locking the observation preprocessing method.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data import PrecipTransform
from src.models.checkpoint_loader import load_model_checkpoint
from src.eval.verification_metrics import mae, rmse, pearson_r, fss


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "checkpoint": (
            data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt"
        ),
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def encode(model, raw, transform, device, batch_size):
    out = []
    for s in range(0, len(raw), batch_size):
        batch = torch.stack(
            [transform(x) for x in raw[s:s+batch_size]], dim=0
        ).to(device, non_blocking=True)
        with torch.no_grad():
            _, z = model(batch)
        out.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def spearman(x, y):
    a = pd.Series(x).rank(method="average")
    b = pd.Series(y).rank(method="average")
    return float(a.corr(b))


def quantiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    qs = [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1]
    vals = np.quantile(x, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default=str(d["archive"]))
    p.add_argument("--checkpoint", default=str(d["checkpoint"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--fss-thresholds", type=float, nargs="+", default=[0.1,1.0,5.0,10.0]
    )
    p.add_argument(
        "--fss-widths", type=int, nargs="+", default=[1,7,15,31]
    )
    p.add_argument(
        "--representative-fss-threshold", type=float, default=1.0
    )
    p.add_argument(
        "--representative-fss-width", type=int, default=15
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    archive = Path(args.archive_dir)
    out = archive / "results"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((archive/"archive_metadata.json").read_text())
    if metadata.get("alignment_status","").startswith("REQUIRES"):
        print(
            "WARNING: archive metadata says manual grid alignment/orientation "
            "check has not yet been recorded as complete."
        )

    cases = pd.read_csv(archive/"cases.csv")
    fcst = np.load(archive/"hrrr_qpf6_mm.npy", mmap_mode="r")
    obs_b = np.load(archive/"stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    obs_l = np.load(archive/"stageiv_bilin_qpf6_mm.npy", mmap_mode="r")
    if not (len(cases) == len(fcst) == len(obs_b) == len(obs_l)):
        raise RuntimeError("Archive arrays/cases length mismatch.")

    transform = PrecipTransform.from_stats_json(
        "log1p_zscore", args.stats_json
    )
    device = torch.device("cuda")
    model, ckpt = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()

    zf = encode(model, fcst, transform, device, args.batch_size)
    zb = encode(model, obs_b, transform, device, args.batch_size)
    zl = encode(model, obs_l, transform, device, args.batch_size)

    sim_b = np.sum(zf*zb, axis=1)
    sim_l = np.sum(zf*zl, axis=1)

    records = []
    for i in range(len(cases)):
        row = cases.iloc[i].to_dict()
        f = np.asarray(fcst[i], dtype=np.float32)
        b = np.asarray(obs_b[i], dtype=np.float32)
        l = np.asarray(obs_l[i], dtype=np.float32)

        r = {
            **row,
            "learned_budget": float(sim_b[i]),
            "learned_bilin": float(sim_l[i]),
            "learned_regrid_delta": float(sim_b[i]-sim_l[i]),
            "budget_rmse": rmse(f,b),
            "bilin_rmse": rmse(f,l),
            "budget_mae": mae(f,b),
            "bilin_mae": mae(f,l),
            "budget_corr": pearson_r(f,b),
            "bilin_corr": pearson_r(f,l),
            "obs_budget_bilin_rmse": rmse(b,l),
            "obs_budget_bilin_corr": pearson_r(b,l),
            "fcst_wet_fraction_ge0p1": float(np.mean(f >= 0.1)),
            "obs_budget_wet_fraction_ge0p1": float(np.mean(b >= 0.1)),
            "obs_bilin_wet_fraction_ge0p1": float(np.mean(l >= 0.1)),
            "fcst_mean_mm": float(np.mean(f)),
            "obs_budget_mean_mm": float(np.mean(b)),
            "obs_bilin_mean_mm": float(np.mean(l)),
        }

        for t in args.fss_thresholds:
            for w in args.fss_widths:
                tag = f"t{t:g}_w{w}".replace(".","p")
                r[f"budget_fss_{tag}"] = fss(f,b,t,w)
                r[f"bilin_fss_{tag}"] = fss(f,l,t,w)

        records.append(r)

    df = pd.DataFrame(records)
    per_case_path = out/"per_case_metrics.csv.gz"
    df.to_csv(per_case_path, index=False)

    rep_tag = (
        f"t{args.representative_fss_threshold:g}_"
        f"w{args.representative_fss_width}"
    ).replace(".","p")
    rep_fss = f"budget_fss_{rep_tag}"
    if rep_fss not in df:
        raise RuntimeError(
            f"Representative FSS {rep_fss} was not computed. "
            "Ensure threshold/width are included."
        )

    summary_rows = []
    for col in [
        "learned_budget","learned_bilin","budget_corr","bilin_corr",
        "budget_rmse","bilin_rmse",rep_fss,
    ]:
        q = quantiles(df[col])
        summary_rows.append({
            "metric": col,
            "mean": float(np.nanmean(df[col])),
            "std": float(np.nanstd(df[col])),
            **q,
        })
    pd.DataFrame(summary_rows).to_csv(
        out/"score_summary.csv", index=False
    )

    delta = np.abs(df["learned_regrid_delta"].to_numpy())
    regrid = {
        "n_cases": len(df),
        "checkpoint_step": int(ckpt.get("step",-1)),
        "learned_budget_bilin_pearson": float(
            np.corrcoef(df["learned_budget"], df["learned_bilin"])[0,1]
        ),
        "learned_budget_bilin_spearman": spearman(
            df["learned_budget"], df["learned_bilin"]
        ),
        "learned_abs_delta_median": float(np.nanmedian(delta)),
        "learned_abs_delta_p90": float(np.nanquantile(delta,.90)),
        "learned_abs_delta_p95": float(np.nanquantile(delta,.95)),
        "learned_abs_delta_p99": float(np.nanquantile(delta,.99)),
        "learned_abs_delta_max": float(np.nanmax(delta)),
        "budget_vs_bilin_obs_rmse_median_mm": float(
            np.nanmedian(df["obs_budget_bilin_rmse"])
        ),
        "budget_vs_bilin_obs_corr_median": float(
            np.nanmedian(df["obs_budget_bilin_corr"])
        ),
        "budget_learned_vs_corr_pearson": float(
            pd.Series(df["learned_budget"]).corr(df["budget_corr"])
        ),
        "budget_learned_vs_corr_spearman": spearman(
            df["learned_budget"], df["budget_corr"]
        ),
        "budget_learned_vs_representative_fss_pearson": float(
            pd.Series(df["learned_budget"]).corr(df[rep_fss])
        ),
        "budget_learned_vs_representative_fss_spearman": spearman(
            df["learned_budget"], df[rep_fss]
        ),
        "representative_fss": rep_fss,
    }
    (out/"regrid_and_association_summary.json").write_text(
        json.dumps(regrid, indent=2) + "\n"
    )

    # Ranking files for quick meteorological inspection.
    df.sort_values("learned_budget", ascending=False).head(50).to_csv(
        out/"highest_learned_similarity.csv", index=False
    )
    df.sort_values("learned_budget", ascending=True).head(50).to_csv(
        out/"lowest_learned_similarity.csv", index=False
    )

    # Rank disagreements are more robust than comparing raw metric magnitudes.
    df2 = df.copy()
    df2["learned_rank_pct"] = df2["learned_budget"].rank(pct=True)
    df2["fss_rank_pct"] = df2[rep_fss].rank(pct=True)
    df2["corr_rank_pct"] = df2["budget_corr"].rank(pct=True)
    df2["learned_minus_fss_rank"] = (
        df2["learned_rank_pct"] - df2["fss_rank_pct"]
    )
    df2["learned_minus_corr_rank"] = (
        df2["learned_rank_pct"] - df2["corr_rank_pct"]
    )
    df2.reindex(
        df2["learned_minus_fss_rank"].abs().sort_values(
            ascending=False
        ).index
    ).head(100).to_csv(
        out/"largest_learned_fss_rank_disagreements.csv", index=False
    )
    df2.reindex(
        df2["learned_minus_corr_rank"].abs().sort_values(
            ascending=False
        ).index
    ).head(100).to_csv(
        out/"largest_learned_corr_rank_disagreements.csv", index=False
    )

    # Diagnostic plots.
    plt.figure(figsize=(8,5))
    plt.hist(df["learned_budget"], bins=40)
    plt.xlabel("S(F,O): learned cosine similarity (Stage IV BUDGET)")
    plt.ylabel("case count")
    plt.title("HRRR–Stage IV learned similarity distribution")
    savefig(plots/"learned_similarity_histogram.png")

    plt.figure(figsize=(6,6))
    plt.scatter(df["learned_bilin"], df["learned_budget"], s=10, alpha=.5)
    lo = min(df["learned_bilin"].min(), df["learned_budget"].min())
    hi = max(df["learned_bilin"].max(), df["learned_budget"].max())
    plt.plot([lo,hi],[lo,hi],"--",linewidth=1)
    plt.xlabel("Learned similarity: BILIN Stage IV")
    plt.ylabel("Learned similarity: BUDGET Stage IV")
    plt.title("Stage-IV regridding sensitivity")
    savefig(plots/"budget_vs_bilin_learned_similarity.png")

    plt.figure(figsize=(7,5.5))
    plt.scatter(df["budget_corr"], df["learned_budget"], s=10, alpha=.5)
    plt.xlabel("Pearson correlation")
    plt.ylabel("Learned similarity")
    plt.title("Learned similarity vs correlation")
    savefig(plots/"learned_vs_correlation.png")

    plt.figure(figsize=(7,5.5))
    plt.scatter(df[rep_fss], df["learned_budget"], s=10, alpha=.5)
    plt.xlabel(rep_fss)
    plt.ylabel("Learned similarity")
    plt.title("Learned similarity vs representative FSS")
    savefig(plots/"learned_vs_fss.png")

    print("="*84)
    print("STAGE-IV REFERENCE EVALUATION COMPLETE")
    print("="*84)
    print(f"cases                    : {len(df)}")
    print(f"checkpoint step          : {ckpt.get('step',-1)}")
    print(
        "learned BUDGET quantiles: "
        + ", ".join(
            f"{k}={v:.3f}" for k,v in quantiles(df["learned_budget"]).items()
        )
    )
    print(
        f"|BUDGET-BILIN| median/p95/max: "
        f"{regrid['learned_abs_delta_median']:.4f} / "
        f"{regrid['learned_abs_delta_p95']:.4f} / "
        f"{regrid['learned_abs_delta_max']:.4f}"
    )
    print(
        f"Spearman learned vs corr : "
        f"{regrid['budget_learned_vs_corr_spearman']:.3f}"
    )
    print(
        f"Spearman learned vs FSS  : "
        f"{regrid['budget_learned_vs_representative_fss_spearman']:.3f}"
    )
    print(f"results                  : {out}")


if __name__ == "__main__":
    main()

