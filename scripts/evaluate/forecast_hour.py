#!/usr/bin/env python3
"""Evaluate SYN-3, SYN-8, and NFV by HRRR forecast hour against Stage IV.

This is the learned-metric GPU stage for the canonical forecast-hour diagnostic.
Traditional metrics are computed separately by forecast_hour_traditional.py and
merged here into one score table for plotting.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PrecipTransform
from src.eval.final_models import DISPLAY_NAMES, checkpoint_paths, normalization_stats
from src.models.checkpoint_loader import load_model_checkpoint


def defaults() -> dict[str, Path]:
    data = Path(os.environ["DATA_DIR"])
    return {
        "archive": data / "derived/evaluation/stageiv_forecast_hour_v1",
        "stats": normalization_stats(data),
    }


def checkpoint_step(ckpt: dict) -> int:
    for key in ("fine_tune_step", "step", "global_step"):
        if key in ckpt:
            try:
                return int(ckpt[key])
            except Exception:
                pass
    return -1


@torch.inference_mode()
def encode(model, raw: np.ndarray, transform, device: torch.device, batch_size: int) -> np.ndarray:
    pieces = []
    for start in range(0, len(raw), batch_size):
        end = min(len(raw), start + batch_size)
        x = torch.stack([transform(v) for v in raw[start:end]], dim=0).to(device, non_blocking=True)
        _, z = model(x)
        pieces.append(F.normalize(z.float(), dim=-1).cpu().numpy())
        if start == 0 or end == len(raw) or (start // batch_size) % 50 == 0:
            print(f"    encoded {end:6d}/{len(raw):6d}", flush=True)
    return np.concatenate(pieces, axis=0)


def infer_one(name: str, checkpoint: Path, fcst, obs, transform, device, batch_size: int):
    if not checkpoint.is_file():
        raise RuntimeError(f"{DISPLAY_NAMES[name]} checkpoint not found: {checkpoint}")
    print(f"\nLoading {DISPLAY_NAMES[name]}: {checkpoint}", flush=True)
    model, ckpt = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    print("  HRRR embeddings", flush=True)
    zf = encode(model, fcst, transform, device, batch_size)
    print("  Stage-IV embeddings (once per valid time)", flush=True)
    zo = encode(model, obs, transform, device, batch_size)
    step = checkpoint_step(ckpt)
    del model, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return zf, zo, step


def main() -> None:
    d = defaults()
    ckpt_default = checkpoint_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=str(d["archive"]))
    ap.add_argument("--syn3-checkpoint", default=str(ckpt_default["syn3"]))
    ap.add_argument("--syn8-checkpoint", default=str(ckpt_default["syn8"]))
    ap.add_argument("--nfv-checkpoint", default=str(ckpt_default["nfv"]))
    ap.add_argument("--stats-json", default=str(d["stats"]))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for learned forecast-hour evaluation")

    arc = Path(args.archive_dir)
    required = [
        arc / "preflight_report.json",
        arc / "traditional_metrics_by_forecast_hour.csv",
        arc / "cases.csv",
        arc / "valid_times.csv",
        arc / "hrrr_qpf6_mm.npy",
        arc / "stageiv_budget_qpf6_mm.npy",
        Path(args.stats_json),
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Required forecast-hour inputs are missing:\n" + "\n".join(missing))

    preflight = json.loads((arc / "preflight_report.json").read_text())
    if preflight.get("status") != "PASS":
        raise SystemExit("preflight_report.json is not PASS")

    cases = pd.read_csv(arc / "cases.csv")
    traditional = pd.read_csv(arc / "traditional_metrics_by_forecast_hour.csv")
    fcst = np.load(arc / "hrrr_qpf6_mm.npy", mmap_mode="r")
    obs = np.load(arc / "stageiv_budget_qpf6_mm.npy", mmap_mode="r")

    for col in ("case_index", "stageiv_index", "forecast_hour"):
        cases[col] = pd.to_numeric(cases[col], errors="raise").astype(int)
    counts = cases["forecast_hour"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise SystemExit(f"Cases are not balanced by forecast hour:\n{counts}")
    if len(fcst) != len(cases):
        raise SystemExit("HRRR array length != cases.csv")
    if cases["stageiv_index"].max() >= len(obs):
        raise SystemExit("Invalid Stage-IV index in cases.csv")

    transform = PrecipTransform.from_stats_json("log1p_zscore", args.stats_json)
    device = torch.device("cuda")
    obs_idx = cases["stageiv_index"].to_numpy(dtype=int)
    learned_cols = {}
    steps = {}

    paths = {
        "syn3": Path(args.syn3_checkpoint),
        "syn8": Path(args.syn8_checkpoint),
        "nfv": Path(args.nfv_checkpoint),
    }
    for key in ("syn3", "syn8", "nfv"):
        zf, zo, steps[key] = infer_one(
            key, paths[key], fcst, obs, transform, device, args.batch_size
        )
        learned_cols[key] = np.sum(zf * zo[obs_idx], axis=1).astype(np.float32)
        del zf, zo
        gc.collect()

    keep = [
        "case_index", "sample_id", "valid_id", "valid_end", "init_hour",
        "forecast_hour", "stageiv_index", "y0", "x0",
    ]
    keep = [c for c in keep if c in cases.columns]
    per_case = cases[keep].copy()
    for key, values in learned_cols.items():
        per_case[key] = values
    per_case.to_csv(arc / "learned_similarity_per_case.csv.gz", index=False)

    learned = per_case.groupby("forecast_hour", as_index=False).agg(
        n_cases_learned=("case_index", "size"),
        syn3=("syn3", "mean"),
        syn8=("syn8", "mean"),
        nfv=("nfv", "mean"),
    )
    combined = traditional.merge(learned, on="forecast_hour", how="inner", validate="one_to_one")
    if not np.array_equal(combined["n_cases"].to_numpy(), combined["n_cases_learned"].to_numpy()):
        raise RuntimeError("Traditional and learned sample counts disagree by forecast hour")
    combined = combined.drop(columns=["n_cases_learned"])
    combined.to_csv(arc / "verification_scores_by_forecast_hour.csv", index=False)

    meta = {
        "learned_score": "mean raw cosine similarity of L2-normalized projection z embeddings",
        "checkpoints": {k: str(v) for k, v in paths.items()},
        "checkpoint_steps": steps,
        "stats_json": str(args.stats_json),
        "n_cases": int(len(cases)),
        "n_unique_stageiv": int(len(obs)),
    }
    (arc / "learned_metrics_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\nFORECAST-HOUR LEARNED EVALUATION COMPLETE")
    print(combined.to_string(index=False))
    print(f"Output: {arc / 'verification_scores_by_forecast_hour.csv'}")


if __name__ == "__main__":
    main()
