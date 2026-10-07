#!/usr/bin/env python3
"""
Diagnose cross-domain transfer of the frozen GroupNorm precipitation metric.

This script is intentionally diagnostic, not a new model-selection exercise.
It uses the existing HRRR/Stage-IV validation archive and the already-selected
GroupNorm checkpoint.

It tests:
  1) pipeline identities and batch-size invariance;
  2) BUDGET vs BILIN embedding sensitivity;
  3) projection z vs backbone h for HRRR->Stage-IV similarity;
  4) matched HRRR/Stage-IV pairs vs activity-matched random mismatches;
  5) source/domain separation in embedding space;
  6) dependence of score on precipitation amount / wet coverage;
  7) same-source displacement response for HRRR and Stage IV.

Outputs are written under:
  <archive>/results/domain_transfer_diagnostics_v1/
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


def defaults():
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_reference_v1",
        "checkpoint": data / "derived/experiments/groupnorm_pilot_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
    }


def encode_both(model, raw, transform, device, batch_size):
    hs, zs = [], []
    for s in range(0, len(raw), batch_size):
        arr = raw[s:s+batch_size]
        batch = torch.stack([transform(x) for x in arr], dim=0).to(
            device, non_blocking=True
        )
        with torch.no_grad():
            h, z = model(batch)
        hs.append(F.normalize(h.float(), dim=-1).cpu().numpy())
        zs.append(F.normalize(z.float(), dim=-1).cpu().numpy())
    return np.concatenate(hs, axis=0), np.concatenate(zs, axis=0)


def cosine_rows(a, b):
    return np.sum(np.asarray(a) * np.asarray(b), axis=1)


def safe_spearman(x, y):
    a = pd.Series(np.asarray(x, float))
    b = pd.Series(np.asarray(y, float))
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(a[ok].rank(method="average").corr(b[ok].rank(method="average")))


def field_stats(x, wet_threshold=0.1):
    x = np.asarray(x, np.float32)
    wet = x >= wet_threshold
    vals = x[wet]
    return {
        "mean_mm": float(np.mean(x)),
        "max_mm": float(np.max(x)),
        "wet_fraction_ge0p1": float(np.mean(wet)),
        "wet_count_ge0p1": int(np.sum(wet)),
        "p90_wet_ge0p1_mm": float(np.percentile(vals, 90)) if vals.size else np.nan,
    }


def activity_class(wet_fraction):
    # Descriptive bins only; not training/evaluation thresholds.
    if wet_fraction == 0:
        return "dry"
    if wet_fraction < 0.01:
        return "sparse_<1%"
    if wet_fraction < 0.10:
        return "1-10%"
    return "widespread_>=10%"


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def plot_scatter(x, y, xlabel, ylabel, title, path, logx=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(7.2, 5.4))
    plt.scatter(x[ok], y[ok], s=12, alpha=0.45)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.2)
    savefig(path)


def plot_hist(values_a, values_b, label_a, label_b, xlabel, title, path):
    a = np.asarray(values_a, float)
    b = np.asarray(values_b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    bins = np.linspace(-1, 1, 61)
    plt.figure(figsize=(7.2, 5.2))
    plt.hist(a, bins=bins, alpha=0.55, density=True, label=label_a)
    plt.hist(b, bins=bins, alpha=0.55, density=True, label=label_b)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    savefig(path)


def shift_no_wrap(x, dy, dx):
    x = np.asarray(x)
    out = np.zeros_like(x)
    ny, nx = x.shape

    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx)

    dst_y0 = max(0, dy)
    dst_y1 = min(ny, ny + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(nx, nx + dx)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = x[src_y0:src_y1, src_x0:src_x1]
    return out


def pca_2d(x):
    x = np.asarray(x, float)
    xc = x - x.mean(axis=0, keepdims=True)
    # n x d is only ~2000 x 128/512 here.
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T, s


def nearest_centroid_domain_accuracy(hrrr, obs, seed=20261004):
    """
    Case-grouped train/test split. Classifies normalized embeddings by nearest
    source centroid (equivalent to a simple linear source discriminator).
    """
    n = len(hrrr)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(0.7*n))
    tr, te = idx[:cut], idx[cut:]

    ch = F.normalize(torch.tensor(hrrr[tr]).mean(0), dim=0).numpy()
    co = F.normalize(torch.tensor(obs[tr]).mean(0), dim=0).numpy()

    x = np.concatenate([hrrr[te], obs[te]], axis=0)
    y = np.concatenate([np.zeros(len(te), int), np.ones(len(te), int)])
    sh = x @ ch
    so = x @ co
    pred = (so > sh).astype(int)
    acc = float(np.mean(pred == y))
    return acc, float(np.dot(ch, co))


def make_activity_matched_random_indices(obs_wet_fraction, k, seed):
    """
    For each case, select k random Stage-IV cases from the same wet-fraction
    decile where possible. Excludes self. This is a harder control than fully
    random mismatches because precipitation activity is approximately matched.
    """
    x = np.asarray(obs_wet_fraction, float)
    n = len(x)
    # Robust bins: exact dry gets its own bin; positive cases get quantile bins.
    labels = np.full(n, -1, int)
    dry = x == 0
    labels[dry] = 0
    pos = ~dry
    if pos.any():
        ranks = pd.Series(x[pos]).rank(method="first", pct=True).to_numpy()
        labels[pos] = 1 + np.minimum((ranks*10).astype(int), 9)

    rng = np.random.default_rng(seed)
    out = np.empty((n, k), int)
    all_idx = np.arange(n)
    for i in range(n):
        pool = all_idx[(labels == labels[i]) & (all_idx != i)]
        if len(pool) < k:
            pool = all_idx[all_idx != i]
        out[i] = rng.choice(pool, size=k, replace=(len(pool) < k))
    return out


def random_control(emb_f, emb_o, random_idx):
    matched = cosine_rows(emb_f, emb_o)
    sims = np.stack(
        [cosine_rows(emb_f, emb_o[random_idx[:, j]]) for j in range(random_idx.shape[1])],
        axis=1
    )
    hardest = sims.max(axis=1)
    mean_random = sims.mean(axis=1)
    top1 = float(np.mean(matched > hardest) * 100.0)
    pairwise_win = float(np.mean(matched[:, None] > sims) * 100.0)
    return matched, mean_random, hardest, top1, pairwise_win


def displacement_diagnostic(
    model, raw, base_h, base_z, transform, device, batch_size,
    sample_idx, magnitudes, source_name, outdir
):
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    rows = []
    raw_sel = np.asarray(raw[sample_idx])

    for mag in magnitudes:
        if mag == 0:
            sims_h = np.ones(len(sample_idx))
            sims_z = np.ones(len(sample_idx))
            for i, case_idx in enumerate(sample_idx):
                rows.append({
                    "source": source_name, "case_index": int(case_idx),
                    "shift_cells": 0, "direction": "identity",
                    "h_similarity": 1.0, "z_similarity": 1.0,
                })
            continue

        for dy0, dx0 in directions:
            shifted = np.stack(
                [shift_no_wrap(x, dy0*mag, dx0*mag) for x in raw_sel]
            ).astype(np.float32)
            hs, zs = encode_both(model, shifted, transform, device, batch_size)
            sh = cosine_rows(base_h[sample_idx], hs)
            sz = cosine_rows(base_z[sample_idx], zs)
            direction = f"dy{dy0:+d}_dx{dx0:+d}"
            for j, case_idx in enumerate(sample_idx):
                rows.append({
                    "source": source_name, "case_index": int(case_idx),
                    "shift_cells": int(mag), "direction": direction,
                    "h_similarity": float(sh[j]), "z_similarity": float(sz[j]),
                })

    return pd.DataFrame(rows)


def main():
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default=str(d["archive"]))
    p.add_argument("--checkpoint", default=str(d["checkpoint"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batch-invariance-count", type=int, default=64)
    p.add_argument("--random-negatives", type=int, default=20)
    p.add_argument("--transform-sample-count", type=int, default=500)
    p.add_argument("--shift-cells", type=int, nargs="+", default=[0,1,3,6])
    p.add_argument("--seed", type=int, default=20261004)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    archive = Path(args.archive_dir)
    result_dir = archive/"results"/"domain_transfer_diagnostics_v1"
    plot_dir = result_dir/"plots"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(archive/"cases.csv")
    fcst = np.load(archive/"hrrr_qpf6_mm.npy", mmap_mode="r")
    obs_b = np.load(archive/"stageiv_budget_qpf6_mm.npy", mmap_mode="r")
    obs_l = np.load(archive/"stageiv_bilin_qpf6_mm.npy", mmap_mode="r")
    if not (len(cases) == len(fcst) == len(obs_b) == len(obs_l)):
        raise RuntimeError("Archive arrays/cases length mismatch.")

    transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    device = torch.device("cuda")
    model, ckpt = load_model_checkpoint(args.checkpoint, device=device)
    model.eval()

    print("="*92)
    print("STAGE-IV DOMAIN-TRANSFER DIAGNOSTICS")
    print("="*92)
    print(f"cases                 : {len(cases)}")
    print(f"checkpoint            : {args.checkpoint}")
    print(f"checkpoint step       : {ckpt.get('step',-1)}")
    print(f"batch size            : {args.batch_size}")
    print(f"random negatives/case : {args.random_negatives}")
    print()

    # ------------------------------------------------------------------
    # 1) Encode all fields in both backbone h and projection z.
    # ------------------------------------------------------------------
    print("Encoding HRRR...")
    hf, zf = encode_both(model, fcst, transform, device, args.batch_size)
    print("Encoding Stage IV BUDGET...")
    hb, zb = encode_both(model, obs_b, transform, device, args.batch_size)
    print("Encoding Stage IV BILIN...")
    hl, zl = encode_both(model, obs_l, transform, device, args.batch_size)

    # ------------------------------------------------------------------
    # 2) Raw field statistics and basic cross-domain similarities.
    # ------------------------------------------------------------------
    records = []
    for i in range(len(cases)):
        fs = field_stats(fcst[i])
        bs = field_stats(obs_b[i])
        ls = field_stats(obs_l[i])
        r = dict(cases.iloc[i].to_dict())
        for k,v in fs.items(): r[f"fcst_{k}"] = v
        for k,v in bs.items(): r[f"obs_budget_{k}"] = v
        for k,v in ls.items(): r[f"obs_bilin_{k}"] = v
        r.update({
            "z_fo_budget": float(np.dot(zf[i], zb[i])),
            "z_fo_bilin": float(np.dot(zf[i], zl[i])),
            "h_fo_budget": float(np.dot(hf[i], hb[i])),
            "h_fo_bilin": float(np.dot(hf[i], hl[i])),
            "z_budget_bilin": float(np.dot(zb[i], zl[i])),
            "h_budget_bilin": float(np.dot(hb[i], hl[i])),
            "z_ff_identity": float(np.dot(zf[i], zf[i])),
            "z_oo_identity": float(np.dot(zb[i], zb[i])),
            "h_ff_identity": float(np.dot(hf[i], hf[i])),
            "h_oo_identity": float(np.dot(hb[i], hb[i])),
        })
        r["abs_wet_fraction_difference"] = abs(
            fs["wet_fraction_ge0p1"] - bs["wet_fraction_ge0p1"]
        )
        r["abs_mean_mm_difference"] = abs(fs["mean_mm"] - bs["mean_mm"])
        r["obs_activity_class"] = activity_class(bs["wet_fraction_ge0p1"])
        r["both_dry_ge0p1"] = (
            fs["wet_count_ge0p1"] == 0 and bs["wet_count_ge0p1"] == 0
        )
        r["fcst_only_wet_ge0p1"] = (
            fs["wet_count_ge0p1"] > 0 and bs["wet_count_ge0p1"] == 0
        )
        r["obs_only_wet_ge0p1"] = (
            fs["wet_count_ge0p1"] == 0 and bs["wet_count_ge0p1"] > 0
        )
        r["both_wet_ge0p1"] = (
            fs["wet_count_ge0p1"] > 0 and bs["wet_count_ge0p1"] > 0
        )
        records.append(r)
    df = pd.DataFrame(records)

    # Merge original traditional-metric result if available.
    old = archive/"results"/"per_case_metrics.csv.gz"
    if old.exists():
        olddf = pd.read_csv(old)
        join_cols = [
            c for c in ("case_index","sample_id","valid_end","forecast_hour")
            if c in df.columns and c in olddf.columns
        ]
        if join_cols:
            keep = [
                c for c in olddf.columns
                if c not in df.columns or c in join_cols
            ]
            df = df.merge(olddf[keep], on=join_cols, how="left")

    # ------------------------------------------------------------------
    # 3) Batch-size invariance: GroupNorm should not depend on companions.
    # ------------------------------------------------------------------
    nbi = min(args.batch_invariance_count, len(cases))
    subset = np.asarray(fcst[:nbi])
    h1, z1 = encode_both(model, subset, transform, device, 1)
    hB, zB = encode_both(model, subset, transform, device, args.batch_size)
    batch_inv = {
        "n_cases": int(nbi),
        "z_max_abs_component_difference": float(np.max(np.abs(z1-zB))),
        "h_max_abs_component_difference": float(np.max(np.abs(h1-hB))),
        "z_cosine_min_between_batchings": float(np.min(cosine_rows(z1,zB))),
        "h_cosine_min_between_batchings": float(np.min(cosine_rows(h1,hB))),
    }

    # ------------------------------------------------------------------
    # 4) Activity-matched random cross-domain controls.
    # ------------------------------------------------------------------
    ridx = make_activity_matched_random_indices(
        df["obs_budget_wet_fraction_ge0p1"].to_numpy(),
        args.random_negatives,
        args.seed,
    )
    zm, zrmean, zrhard, ztop1, zpairwin = random_control(zf, zb, ridx)
    hm, hrmean, hrhard, htop1, hpairwin = random_control(hf, hb, ridx)
    df["z_random_activitymatched_mean"] = zrmean
    df["z_random_activitymatched_hardest"] = zrhard
    df["z_matched_minus_randommean"] = zm-zrmean
    df["z_matched_minus_randomhardest"] = zm-zrhard
    df["h_random_activitymatched_mean"] = hrmean
    df["h_random_activitymatched_hardest"] = hrhard
    df["h_matched_minus_randommean"] = hm-hrmean
    df["h_matched_minus_randomhardest"] = hm-hrhard

    # ------------------------------------------------------------------
    # 5) Domain/source separation in h and z.
    # ------------------------------------------------------------------
    z_source_acc, z_centroid_cos = nearest_centroid_domain_accuracy(zf, zb, args.seed)
    h_source_acc, h_centroid_cos = nearest_centroid_domain_accuracy(hf, hb, args.seed)

    zpc, zsing = pca_2d(np.concatenate([zf,zb], axis=0))
    hpc, hsing = pca_2d(np.concatenate([hf,hb], axis=0))
    n = len(df)
    pca_df = pd.DataFrame({
        "source": ["HRRR"]*n + ["StageIV"]*n,
        "case_index": list(range(n))*2,
        "z_pc1": zpc[:,0],
        "z_pc2": zpc[:,1],
        "h_pc1": hpc[:,0],
        "h_pc2": hpc[:,1],
    })
    pca_df.to_csv(result_dir/"embedding_pca.csv", index=False)

    # ------------------------------------------------------------------
    # 6) Same-source displacement response.
    # Use both-wet cases so dry identity-like fields do not dominate.
    # ------------------------------------------------------------------
    eligible = np.where(df["both_wet_ge0p1"].to_numpy(bool))[0]
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.transform_sample_count:
        sample_idx = np.sort(
            rng.choice(eligible, args.transform_sample_count, replace=False)
        )
    else:
        sample_idx = eligible
    print(f"Displacement diagnostic cases: {len(sample_idx)} both-wet pairs")

    disp_hrrr = displacement_diagnostic(
        model, fcst, hf, zf, transform, device, args.batch_size,
        sample_idx, args.shift_cells, "HRRR", result_dir
    )
    disp_obs = displacement_diagnostic(
        model, obs_b, hb, zb, transform, device, args.batch_size,
        sample_idx, args.shift_cells, "StageIV", result_dir
    )
    disp = pd.concat([disp_hrrr,disp_obs], ignore_index=True)
    disp.to_csv(result_dir/"same_source_displacement.csv.gz", index=False)

    # ------------------------------------------------------------------
    # 7) Summary tables.
    # ------------------------------------------------------------------
    df.to_csv(result_dir/"per_case_domain_diagnostics.csv.gz", index=False)

    identity = {
        "z_ff_min": float(df["z_ff_identity"].min()),
        "z_oo_min": float(df["z_oo_identity"].min()),
        "h_ff_min": float(df["h_ff_identity"].min()),
        "h_oo_min": float(df["h_oo_identity"].min()),
        "z_budget_bilin_median": float(df["z_budget_bilin"].median()),
        "z_budget_bilin_p01": float(df["z_budget_bilin"].quantile(.01)),
        "z_budget_bilin_min": float(df["z_budget_bilin"].min()),
        "h_budget_bilin_median": float(df["h_budget_bilin"].median()),
        "h_budget_bilin_p01": float(df["h_budget_bilin"].quantile(.01)),
        "h_budget_bilin_min": float(df["h_budget_bilin"].min()),
        **batch_inv,
    }

    source_shift = {
        "z_source_nearest_centroid_accuracy_pct": 100*z_source_acc,
        "h_source_nearest_centroid_accuracy_pct": 100*h_source_acc,
        "z_source_centroid_cosine": z_centroid_cos,
        "h_source_centroid_cosine": h_centroid_cos,
        "z_matched_top1_vs_activitymatched_random_pct": ztop1,
        "h_matched_top1_vs_activitymatched_random_pct": htop1,
        "z_pairwise_win_vs_activitymatched_random_pct": zpairwin,
        "h_pairwise_win_vs_activitymatched_random_pct": hpairwin,
        "z_matched_median": float(np.median(zm)),
        "h_matched_median": float(np.median(hm)),
        "z_random_mean_median": float(np.median(zrmean)),
        "h_random_mean_median": float(np.median(hrmean)),
    }

    associations = {}
    predictors = [
        "obs_budget_mean_mm",
        "obs_budget_max_mm",
        "obs_budget_wet_fraction_ge0p1",
        "fcst_mean_mm",
        "fcst_max_mm",
        "fcst_wet_fraction_ge0p1",
        "abs_wet_fraction_difference",
        "abs_mean_mm_difference",
    ]
    for space in ("z","h"):
        score = df[f"{space}_fo_budget"]
        for col in predictors:
            associations[f"{space}_score_vs_{col}_spearman"] = safe_spearman(
                score, df[col]
            )

    # activity-class summary
    activity_summary = (
        df.groupby("obs_activity_class", dropna=False)
          .agg(
              n=("z_fo_budget","size"),
              z_mean=("z_fo_budget","mean"),
              z_median=("z_fo_budget","median"),
              z_q10=("z_fo_budget",lambda x: x.quantile(.10)),
              z_q90=("z_fo_budget",lambda x: x.quantile(.90)),
              h_mean=("h_fo_budget","mean"),
              h_median=("h_fo_budget","median"),
              obs_mean_mm=("obs_budget_mean_mm","mean"),
              obs_wet_fraction=("obs_budget_wet_fraction_ge0p1","mean"),
          )
          .reset_index()
    )
    activity_summary.to_csv(result_dir/"score_by_obs_activity_class.csv", index=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step",-1)),
        "n_cases": len(df),
        "identity_and_batch_invariance": identity,
        "source_shift_and_random_controls": source_shift,
        "score_associations": associations,
        "displacement_n_cases": int(len(sample_idx)),
        "interpretation_notes": [
            "Negative cosine is mathematically permitted; the question is whether rankings/generalization are meteorologically meaningful.",
            "Nearest-centroid source accuracy near 50% means little linear source separation; values approaching 100% indicate strong source identity in the embedding.",
            "Matched-vs-random controls use Stage-IV wet-coverage-matched random observations and are diagnostic, not an operational retrieval metric.",
            "Same-source displacement uses zero-filled shifts without wrap and excludes pairs where either original field is completely dry at 0.1 mm.",
        ],
    }
    (result_dir/"diagnostic_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    # ------------------------------------------------------------------
    # 8) Plots: h vs z, dryness/activity, domain shift, random controls.
    # ------------------------------------------------------------------
    plot_scatter(
        df["z_fo_budget"], df["h_fo_budget"],
        "projection z similarity", "backbone h similarity",
        "HRRR–Stage IV: projection z vs backbone h",
        plot_dir/"z_vs_h_similarity.png"
    )

    # Histogram, all vs both-wet
    bothwet = df["both_wet_ge0p1"].to_numpy(bool)
    plt.figure(figsize=(7.2,5.2))
    plt.hist(df["z_fo_budget"], bins=np.linspace(-1,1,61), alpha=.55, density=True, label="all cases")
    plt.hist(df.loc[bothwet,"z_fo_budget"], bins=np.linspace(-1,1,61), alpha=.55, density=True, label="both wet >=0.1 mm")
    plt.xlabel("projection z cosine similarity")
    plt.ylabel("density")
    plt.title("HRRR–Stage IV score distribution: all vs both-wet")
    plt.legend()
    plt.grid(alpha=.2)
    savefig(plot_dir/"z_distribution_all_vs_bothwet.png")

    plot_scatter(
        100*df["obs_budget_wet_fraction_ge0p1"],
        df["z_fo_budget"],
        "Stage IV wet coverage >=0.1 mm (%)",
        "projection z similarity",
        "Learned similarity vs Stage IV precipitation coverage",
        plot_dir/"z_vs_obs_wet_coverage.png",
    )
    plot_scatter(
        df["obs_budget_max_mm"] + 1e-3,
        df["z_fo_budget"],
        "Stage IV maximum QPF6 (mm; log axis)",
        "projection z similarity",
        "Learned similarity vs Stage IV maximum intensity",
        plot_dir/"z_vs_obs_max.png",
        logx=True,
    )
    plot_scatter(
        df["abs_wet_fraction_difference"],
        df["z_fo_budget"],
        "|HRRR wet fraction - Stage IV wet fraction|",
        "projection z similarity",
        "Learned similarity vs precipitation-coverage mismatch",
        plot_dir/"z_vs_wet_fraction_mismatch.png",
    )
    plot_scatter(
        df["abs_mean_mm_difference"] + 1e-5,
        df["z_fo_budget"],
        "|HRRR mean QPF6 - Stage IV mean QPF6| (mm; log axis)",
        "projection z similarity",
        "Learned similarity vs domain-mean precipitation mismatch",
        plot_dir/"z_vs_mean_intensity_mismatch.png",
        logx=True,
    )

    plot_hist(
        zm, zrmean,
        "matched F/O", "activity-matched random F/O",
        "projection z cosine",
        "Projection z: matched vs activity-matched random observations",
        plot_dir/"z_matched_vs_random.png",
    )
    plot_hist(
        hm, hrmean,
        "matched F/O", "activity-matched random F/O",
        "backbone h cosine",
        "Backbone h: matched vs activity-matched random observations",
        plot_dir/"h_matched_vs_random.png",
    )

    # PCA plots.
    plt.figure(figsize=(7,5.5))
    plt.scatter(zpc[:n,0], zpc[:n,1], s=10, alpha=.4, label="HRRR")
    plt.scatter(zpc[n:,0], zpc[n:,1], s=10, alpha=.4, label="Stage IV")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Projection z embedding space by source")
    plt.legend(); plt.grid(alpha=.2)
    savefig(plot_dir/"z_pca_by_source.png")

    plt.figure(figsize=(7,5.5))
    plt.scatter(hpc[:n,0], hpc[:n,1], s=10, alpha=.4, label="HRRR")
    plt.scatter(hpc[n:,0], hpc[n:,1], s=10, alpha=.4, label="Stage IV")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Backbone h embedding space by source")
    plt.legend(); plt.grid(alpha=.2)
    savefig(plot_dir/"h_pca_by_source.png")

    # Same-source displacement summary + plots.
    disp_summary = (
        disp.groupby(["source","shift_cells"])
            .agg(
                n=("z_similarity","size"),
                z_mean=("z_similarity","mean"),
                z_median=("z_similarity","median"),
                z_q25=("z_similarity",lambda x:x.quantile(.25)),
                z_q75=("z_similarity",lambda x:x.quantile(.75)),
                h_mean=("h_similarity","mean"),
                h_median=("h_similarity","median"),
                h_q25=("h_similarity",lambda x:x.quantile(.25)),
                h_q75=("h_similarity",lambda x:x.quantile(.75)),
            )
            .reset_index()
    )
    disp_summary.to_csv(result_dir/"same_source_displacement_summary.csv", index=False)

    for space in ("z","h"):
        plt.figure(figsize=(7.2,5.3))
        for source in ("HRRR","StageIV"):
            q = disp_summary[disp_summary["source"]==source]
            x = q["shift_cells"].to_numpy()
            med = q[f"{space}_median"].to_numpy()
            lo = q[f"{space}_q25"].to_numpy()
            hi = q[f"{space}_q75"].to_numpy()
            plt.plot(x, med, marker="o", label=source)
            plt.fill_between(x, lo, hi, alpha=.18)
        plt.xlabel("displacement magnitude (grid cells)")
        plt.ylabel(f"{space} cosine similarity to unshifted field")
        plt.title(f"Same-source displacement response ({space})")
        plt.legend(); plt.grid(alpha=.2)
        savefig(plot_dir/f"{space}_same_source_displacement.png")

    # Print concise results.
    print("\n" + "-"*92)
    print("IDENTITY / INVARIANCE")
    print("-"*92)
    for k,v in identity.items():
        print(f"{k:42s}: {v:.8f}" if isinstance(v,float) else f"{k:42s}: {v}")
    print("\n" + "-"*92)
    print("CROSS-DOMAIN / SOURCE SHIFT")
    print("-"*92)
    for k,v in source_shift.items():
        print(f"{k:50s}: {v:.4f}")
    print("\n" + "-"*92)
    print("SCORE ASSOCIATIONS")
    print("-"*92)
    for k,v in associations.items():
        print(f"{k:58s}: {v:+.3f}" if np.isfinite(v) else f"{k:58s}: nan")
    print("\nOutputs:", result_dir)
    print("="*92)


if __name__ == "__main__":
    main()

