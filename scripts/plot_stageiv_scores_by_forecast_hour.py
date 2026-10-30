#!/usr/bin/env python3
"""Compute V2/V3 Stage-IV similarities and make forecast-hour verification plot.

All cheap structural checks and traditional verification are expected to have
completed before this GPU stage.  Stage-IV patches are embedded once per valid
time and reused for all eight HRRR leads verifying against that analysis.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.checkpoint_loader import load_model_checkpoint


def defaults() -> dict[str, Path]:
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_forecast_hour_v1",
        "v2": data / "derived/experiments/hard_ranking_finetune_v1/checkpoint_best.pt",
        "v3": data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt",
        "stats": data / "metadata/dataset_v1/precip_normalization_stats_v1.json",
        "out": data / "derived/experiments/hard_ranking_v3_full_v1/evaluation/stageiv_by_forecast_hour_v1",
    }


def load_log1p_stats(path: str | Path) -> tuple[float, float]:
    spec = json.loads(Path(path).read_text())
    if "log1p" not in spec:
        raise RuntimeError(f"{path} has no 'log1p' normalization entry")
    mean = float(spec["log1p"]["mean"])
    std = float(spec["log1p"]["std"])
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid log1p normalization std={std}")
    return mean, std


@torch.inference_mode()
def encode_array(model, arr: np.ndarray, mean: float, std: float, device: torch.device, batch_size: int) -> np.ndarray:
    pieces = []
    for s in range(0, len(arr), batch_size):
        e = min(len(arr), s + batch_size)
        raw = np.array(arr[s:e], dtype=np.float32, copy=True)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Non-finite precipitation encountered in rows {s}:{e}")
        if float(raw.min()) < -1e-5:
            raise RuntimeError(f"Materially negative precipitation encountered in rows {s}:{e}")
        x = torch.from_numpy(raw).unsqueeze(1).to(device, non_blocking=True)
        x.clamp_min_(0.0)
        x.log1p_()
        x.sub_(mean).div_(std)
        _, z = model(x)
        pieces.append(F.normalize(z.float(), dim=-1).cpu().numpy())
        if s == 0 or e == len(arr) or (s // batch_size) % 50 == 0:
            print(f"    encoded {e:6d}/{len(arr):6d}", flush=True)
    return np.concatenate(pieces, axis=0)


def checkpoint_step(ckpt: dict) -> int:
    for key in ["fine_tune_step", "step", "global_step"]:
        if key in ckpt:
            try:
                return int(ckpt[key])
            except Exception:
                pass
    return -1


def infer_one_model(name: str, checkpoint: Path, fcst: np.ndarray, obs: np.ndarray, mean: float, std: float, device, batch_size: int):
    if not checkpoint.is_file():
        raise RuntimeError(f"{name} checkpoint not found: {checkpoint}")
    print(f"\nLoading {name}: {checkpoint}", flush=True)
    model, ckpt = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    print(f"  {name} HRRR embeddings", flush=True)
    zf = encode_array(model, fcst, mean, std, device, batch_size)
    print(f"  {name} Stage-IV embeddings (once per valid time)", flush=True)
    zo = encode_array(model, obs, mean, std, device, batch_size)
    step = checkpoint_step(ckpt)
    del model, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return zf, zo, step


def make_plot(df: pd.DataFrame, png: Path, pdf: Path) -> None:
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(14.5, 8.5), constrained_layout=True)
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.28, linewidth=0.7)
    x = df["forecast_hour"].to_numpy(dtype=int)

    styles = [
        ("ets_0p1", "ETS (0.1 mm)", "#006400", "-", 2.3),
        ("pod_0p1", "POD (0.1 mm)", "#e69500", "-", 2.1),
        ("success_ratio_0p1", "Normalized FAR (Success Ratio)", "#66c266", "-", 2.1),
        ("fss_0p1_w10", "FSS-10", "#08306b", "-", 2.4),
        ("fss_0p1_w5", "FSS-5", "#6baed6", "-", 2.2),
        ("learned_v3", "V3 learned similarity", "black", "-", 2.6),
        ("learned_v2", "V2 learned similarity", "black", ":", 2.6),
    ]

    plotted_cols = []
    for col, label, color, ls, lw in styles:
        if col not in df.columns:
            raise RuntimeError(f"Missing required plotting column {col}")
        y = df[col].to_numpy(dtype=float)
        plotted_cols.append(y)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=6, label=label)

    all_y = np.concatenate([v[np.isfinite(v)] for v in plotted_cols if np.isfinite(v).any()])
    ymin = min(-0.05, float(np.min(all_y)) - 0.04)
    ax.set_ylim(ymin, 1.04)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xticks(x)
    ax.set_xlabel("Forecast Lead Hour")
    ax.set_ylabel("Verification Score")
    ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_title(
        "HRRR 6-h Precipitation Verification by Forecast Hour\n"
        "Validation period: 2024-10-01 to 2025-09-20 | Stage IV reference | 00/06/12/18Z cycles",
        fontweight="bold",
    )

    # Emulate the operational figure's sample-count row.
    count_y = 1.012
    ax.text(x.min() - 0.8, count_y + 0.008, "#SAMPLES", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", color="dimgray", fontsize=10, fontweight="bold", clip_on=False)
    for xi, n in zip(x, df["n_cases"].to_numpy(dtype=int)):
        ax.text(xi, count_y, str(n), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color="dimgray", fontsize=10, fontweight="bold", clip_on=False)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=True, fontsize=10.5, handlelength=3.0, columnspacing=1.3)

    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", default=str(d["archive"]))
    p.add_argument("--v2-checkpoint", default=str(d["v2"]))
    p.add_argument("--v3-checkpoint", default=str(d["v3"]))
    p.add_argument("--stats-json", default=str(d["stats"]))
    p.add_argument("--output-dir", default=str(d["out"]))
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for V2/V3 inference")

    arc = Path(args.archive_dir)
    required = [
        arc / "preflight_report.json",
        arc / "traditional_metrics_by_forecast_hour.csv",
        arc / "cases.csv",
        arc / "valid_times.csv",
        arc / "hrrr_qpf6_mm.npy",
        arc / "stageiv_budget_qpf6_mm.npy",
    ]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise SystemExit(
            "GPU stage refused to start because CPU/preflight outputs are missing:\n"
            + "\n".join(missing)
        )

    preflight = json.loads((arc / "preflight_report.json").read_text())
    if preflight.get("status") != "PASS":
        raise SystemExit("GPU stage refused to start: preflight_report.json is not PASS")

    cases = pd.read_csv(arc / "cases.csv")
    traditional = pd.read_csv(arc / "traditional_metrics_by_forecast_hour.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    cases["case_index"] = pd.to_numeric(cases["case_index"], errors="raise").astype(int)
    cases["stageiv_index"] = pd.to_numeric(cases["stageiv_index"], errors="raise").astype(int)
    cases["forecast_hour"] = pd.to_numeric(cases["forecast_hour"], errors="raise").astype(int)

    # This cheap check intentionally occurs before any checkpoint enters the GPU.
    counts = cases["forecast_hour"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise SystemExit(f"GPU preflight: cases are not balanced by forecast hour:\n{counts}")
    if len(fcst) != len(cases):
        raise SystemExit("GPU preflight: HRRR array length != cases.csv")
    if cases["stageiv_index"].max() >= len(obs):
        raise SystemExit("GPU preflight: invalid Stage-IV index")

    out = Path(args.output_dir)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    mean, std = load_log1p_stats(args.stats_json)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Cases: {len(cases):,}; unique Stage-IV patches: {len(obs):,}", flush=True)

    zf2, zo2, v2_step = infer_one_model(
        "V2", Path(args.v2_checkpoint), fcst, obs, mean, std, device, args.batch_size
    )
    zf3, zo3, v3_step = infer_one_model(
        "V3", Path(args.v3_checkpoint), fcst, obs, mean, std, device, args.batch_size
    )

    obs_idx = cases["stageiv_index"].to_numpy(dtype=int)
    sim2 = np.sum(zf2 * zo2[obs_idx], axis=1)
    sim3 = np.sum(zf3 * zo3[obs_idx], axis=1)

    per_case = cases[["case_index", "sample_id", "valid_id", "valid_end", "init_hour", "forecast_hour", "stageiv_index", "y0", "x0"]].copy()
    per_case["learned_v2"] = sim2
    per_case["learned_v3"] = sim3
    per_case.to_csv(out / "per_case_learned_similarity.csv.gz", index=False)

    learned = (
        per_case.groupby("forecast_hour", as_index=False)
        .agg(
            n_cases_learned=("case_index", "size"),
            learned_v2=("learned_v2", "mean"),
            learned_v3=("learned_v3", "mean"),
            learned_v2_std=("learned_v2", "std"),
            learned_v3_std=("learned_v3", "std"),
        )
        .sort_values("forecast_hour")
    )
    learned.to_csv(out / "learned_metrics_by_forecast_hour.csv", index=False)

    summary = traditional.merge(learned, on="forecast_hour", how="inner", validate="one_to_one")
    if not np.array_equal(summary["n_cases"].to_numpy(), summary["n_cases_learned"].to_numpy()):
        raise RuntimeError("Traditional and learned case counts disagree")
    summary.to_csv(out / "forecast_hour_summary.csv", index=False)

    png = plots / "hrrr_stageiv_verification_scores_by_forecast_hour.png"
    pdf = plots / "hrrr_stageiv_verification_scores_by_forecast_hour.pdf"
    make_plot(summary, png, pdf)

    meta = {
        "archive_dir": str(arc),
        "v2_checkpoint": str(args.v2_checkpoint),
        "v2_step": v2_step,
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_step": v3_step,
        "stats_json": str(args.stats_json),
        "normalization": "log1p_zscore",
        "stageiv_embedding_strategy": "one embedding per unique valid-time patch, reused across leads",
        "n_cases": int(len(cases)),
        "n_unique_stageiv": int(len(obs)),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("=" * 88)
    print("FORECAST-HOUR VERIFICATION COMPLETE")
    print("=" * 88)
    print(summary[["forecast_hour", "n_cases", "ets_0p1", "pod_0p1", "success_ratio_0p1", "fss_0p1_w5", "fss_0p1_w10", "learned_v2", "learned_v3"]].to_string(index=False))
    print(f"\nSummary CSV : {out / 'forecast_hour_summary.csv'}")
    print(f"Figure      : {png}")


if __name__ == "__main__":
    main()
