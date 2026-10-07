#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {path}")


def pick_col(df: pd.DataFrame, aliases, required=True):
    for c in aliases:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find any of these columns: {aliases}")
    return None


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def take_unique(df: pd.DataFrame, n: int, used: set, id_col: str) -> pd.DataFrame:
    keep_rows = []
    for _, row in df.iterrows():
        cid = row[id_col]
        if cid in used:
            continue
        keep_rows.append(row)
        used.add(cid)
        if len(keep_rows) >= n:
            break
    if not keep_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(keep_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Per-case Stage-IV comparison table (csv/parquet)")
    ap.add_argument("--output", required=True, help="Output selected-case manifest CSV")
    ap.add_argument("--n-per-bucket", type=int, default=6)
    ap.add_argument("--wet-total-thresh-in", type=float, default=0.10,
                    help="Minimum domain-summed precip (in) for both-wet filter if wet flags absent")
    ap.add_argument("--min-wet-pixels", type=int, default=16,
                    help="Minimum wet pixels for both-wet filter if wet-pixel columns exist")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(inp).copy()

    # Flexible column lookup
    id_col = pick_col(df, ["case_id", "sample_id", "patch_id", "id"])
    fcst_path_col = pick_col(df, ["forecast_path", "fcst_path", "hrrr_path"], required=False)
    obs_path_col = pick_col(df, ["obs_path", "stageiv_path", "o_path"], required=False)

    v1_col = pick_col(df, ["v1_similarity", "learned_v1", "baseline_v1", "sim_v1"])
    v2_col = pick_col(df, ["v2_similarity", "learned_v2", "hard_ranking_v2", "sim_v2"])
    fss5_col = pick_col(df, ["fss5", "fss_5", "fss_w5"], required=False)
    fss10_col = pick_col(df, ["fss10", "fss_10", "fss_w10"], required=False)
    ets_col = pick_col(df, ["ets_0p1", "ets", "ets01", "ts_0p1"], required=False)
    corr_col = pick_col(df, ["corr", "correlation", "pearson_r"], required=False)

    init_col = pick_col(df, ["init_time", "init", "cycle_time"], required=False)
    valid_col = pick_col(df, ["valid_time", "valid"], required=False)
    lead_col = pick_col(df, ["lead_hours", "fhr", "forecast_hour"], required=False)

    # Build both-wet filter
    both_wet_col = pick_col(df, ["both_wet", "is_both_wet"], required=False)
    if both_wet_col is not None:
        both_wet = df[both_wet_col].astype(bool)
    else:
        fcst_sum_col = pick_col(df, ["forecast_sum_in", "fcst_sum_in", "hrrr_sum_in"], required=False)
        obs_sum_col = pick_col(df, ["obs_sum_in", "stageiv_sum_in", "o_sum_in"], required=False)
        fcst_wetpix_col = pick_col(df, ["forecast_wet_pixels", "fcst_wet_pixels", "hrrr_wet_pixels"], required=False)
        obs_wetpix_col = pick_col(df, ["obs_wet_pixels", "stageiv_wet_pixels", "o_wet_pixels"], required=False)

        if fcst_wetpix_col is not None and obs_wetpix_col is not None:
            both_wet = (
                (pd.to_numeric(df[fcst_wetpix_col], errors="coerce") >= args.min_wet_pixels) &
                (pd.to_numeric(df[obs_wetpix_col], errors="coerce") >= args.min_wet_pixels)
            )
        elif fcst_sum_col is not None and obs_sum_col is not None:
            both_wet = (
                (pd.to_numeric(df[fcst_sum_col], errors="coerce") >= args.wet_total_thresh_in) &
                (pd.to_numeric(df[obs_sum_col], errors="coerce") >= args.wet_total_thresh_in)
            )
        else:
            raise ValueError(
                "Could not determine both-wet cases. Need either both_wet column, "
                "wet-pixel columns, or domain-sum precip columns."
            )

    df["both_wet"] = both_wet
    df["delta_v2_v1"] = pd.to_numeric(df[v2_col], errors="coerce") - pd.to_numeric(df[v1_col], errors="coerce")

    bw = df[df["both_wet"]].copy()
    if len(bw) == 0:
        raise ValueError("No both-wet cases found.")

    used = set()
    picked = []

    # -------------------------
    # Bucket A: high-quality successes
    # -------------------------
    bw["A_score"] = zscore(bw[v2_col])
    if fss10_col:
        bw["A_score"] += 0.75 * zscore(bw[fss10_col])
    if ets_col:
        bw["A_score"] += 0.50 * zscore(bw[ets_col])
    if corr_col:
        bw["A_score"] += 0.25 * zscore(bw[corr_col])

    a = bw.sort_values("A_score", ascending=False)
    a = take_unique(a, args.n_per_bucket, used, id_col)
    if len(a):
        a = a.assign(bucket="A", bucket_desc="high_quality_success", bucket_rank=np.arange(1, len(a)+1))
        picked.append(a)

    # -------------------------
    # Bucket B: biggest V2 improvements
    # -------------------------
    b = bw.sort_values("delta_v2_v1", ascending=False)
    b = take_unique(b, args.n_per_bucket, used, id_col)
    if len(b):
        b = b.assign(bucket="B", bucket_desc="largest_v2_improvements", bucket_rank=np.arange(1, len(b)+1))
        picked.append(b)

    # -------------------------
    # Bucket C: biggest V2 degradations
    # -------------------------
    c = bw.sort_values("delta_v2_v1", ascending=True)
    c = take_unique(c, args.n_per_bucket, used, id_col)
    if len(c):
        c = c.assign(bucket="C", bucket_desc="largest_v2_degradations", bucket_rank=np.arange(1, len(c)+1))
        picked.append(c)

    # -------------------------
    # Bucket D: disagreement cases
    # -------------------------
    # Tradition = mainly FSS-10, with ETS/corr optionally blended in.
    trad = pd.Series(np.zeros(len(bw)), index=bw.index)
    n_trad = 0
    if fss10_col:
        trad += zscore(bw[fss10_col]); n_trad += 1
    if ets_col:
        trad += zscore(bw[ets_col]); n_trad += 1
    if corr_col:
        trad += zscore(bw[corr_col]); n_trad += 1
    if n_trad == 0:
        raise ValueError("Need at least one traditional metric (FSS-10, ETS, or corr) for disagreement bucket.")
    trad = trad / n_trad

    bw["v2_z"] = zscore(bw[v2_col])
    bw["trad_z"] = trad
    bw["disagree_mag"] = (bw["v2_z"] - bw["trad_z"]).abs()

    hi_v2_lo_trad = bw.sort_values(["v2_z", "trad_z"], ascending=[False, True]).copy()
    lo_v2_hi_trad = bw.sort_values(["v2_z", "trad_z"], ascending=[True, False]).copy()

    n1 = args.n_per_bucket // 2
    n2 = args.n_per_bucket - n1

    d1 = take_unique(hi_v2_lo_trad, n1, used, id_col)
    d2 = take_unique(lo_v2_hi_trad, n2, used, id_col)
    d = pd.concat([d1, d2], ignore_index=True)

    if len(d):
        d = d.assign(bucket="D", bucket_desc="v2_traditional_disagreement", bucket_rank=np.arange(1, len(d)+1))
        picked.append(d)

    out_df = pd.concat(picked, ignore_index=True)

    # Keep useful columns up front
    front = [
        "bucket", "bucket_desc", "bucket_rank",
        id_col,
        "both_wet",
        v1_col, v2_col, "delta_v2_v1"
    ]
    for c in [fss5_col, fss10_col, ets_col, corr_col, init_col, valid_col, lead_col, fcst_path_col, obs_path_col]:
        if c and c not in front:
            front.append(c)

    remaining = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + remaining]

    out_df.to_csv(out, index=False)

    print("=" * 80)
    print("SELECTED STAGE-IV CASES")
    print("=" * 80)
    print(f"Input table : {inp}")
    print(f"Output file : {out}")
    print(f"Total cases : {len(out_df)}")
    print(out_df.groupby(["bucket", "bucket_desc"]).size())
    print("=" * 80)


if __name__ == "__main__":
    main()
