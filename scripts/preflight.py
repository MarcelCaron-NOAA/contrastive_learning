#!/usr/bin/env python3
"""Validate the local setup and canonical data products before running jobs.

This preflight intentionally uses only the Python standard library so it can
report missing project data without importing the training stack itself.

Examples
--------
python scripts/preflight.py setup
python scripts/preflight.py support
python scripts/preflight.py training --verify-stores
python scripts/preflight.py diagnostics
python scripts/preflight.py all
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Reporter:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        suffix = f" — {detail}" if detail else ""
        print(f"[{status:4s}] {name}{suffix}")

    def file(self, path: Path, label: str | None = None, *, required: bool = True) -> bool:
        ok = path.is_file()
        self.add(PASS if ok else (FAIL if required else WARN), label or str(path), str(path))
        return ok

    def directory(self, path: Path, label: str | None = None, *, required: bool = True) -> bool:
        ok = path.is_dir()
        self.add(PASS if ok else (FAIL if required else WARN), label or str(path), str(path))
        return ok

    @property
    def failed(self) -> int:
        return sum(s == FAIL for s, _, _ in self.rows)

    @property
    def warned(self) -> int:
        return sum(s == WARN for s, _, _ in self.rows)


def env_path(name: str, reporter: Reporter) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        reporter.add(FAIL, f"environment variable {name}", "not set; source load_env.sh")
        return None
    reporter.add(PASS, f"environment variable {name}", value)
    return Path(value)


def check_setup(reporter: Reporter) -> tuple[Path | None, Path | None]:
    print("\n== setup ==")
    project = env_path("PROJECT_DIR", reporter)
    data = env_path("DATA_DIR", reporter)
    env_path("WORK_DIR", reporter)
    env_path("LOG_DIR", reporter)
    env_path("SCRATCH_DIR", reporter)
    env_path("ENV_DIR", reporter)

    if project is not None:
        reporter.file(project / "site_config.sh", "local site_config.sh")
        reporter.file(project / "load_env.sh", "load_env.sh")

    for command, required in (("sbatch", True), ("squeue", False), ("sacct", False), ("htar", False), ("hsi", False)):
        found = shutil.which(command)
        reporter.add(PASS if found else (FAIL if required else WARN), f"command {command}", found or "not found")

    for module, required in (
        ("numpy", True), ("pandas", True), ("scipy", True), ("xarray", True),
        ("zarr", True), ("torch", True), ("matplotlib", True), ("cartopy", False),
    ):
        found = importlib.util.find_spec(module) is not None
        reporter.add(PASS if found else (FAIL if required else WARN), f"Python import {module}")

    return project, data


def support_files(data: Path) -> list[tuple[str, Path]]:
    return [
        ("eligible anchors", data / "metadata/dataset_v1/eligible_anchors_v1.csv.gz"),
        ("negative pool", data / "metadata/dataset_v1/negative_pool_v1.csv.gz"),
        ("normalization statistics", data / "metadata/dataset_v1/precip_normalization_stats_v1.json"),
        ("SYN-3 checkpoint", data / "derived/experiments/synthetic_perturb_syn3_v2/checkpoint_best.pt"),
        ("SYN-8 checkpoint", data / "derived/experiments/synthetic_perturb_syn8_v2/checkpoint_best.pt"),
        ("NFV checkpoint", data / "derived/experiments/hard_ranking_v3_full_v1/checkpoint_best.pt"),
        ("Stage-IV cases", data / "derived/evaluation/stageiv_reference_v1/cases.csv"),
        ("Stage-IV HRRR patches", data / "derived/evaluation/stageiv_reference_v1/hrrr_qpf6_mm.npy"),
        ("Stage-IV observed patches", data / "derived/evaluation/stageiv_reference_v1/stageiv_budget_qpf6_mm.npy"),
        ("morphology cases", data / "derived/evaluation/wang_transform_v1/cases.csv"),
        ("morphology original patches", data / "derived/evaluation/wang_transform_v1/original_patches.npz"),
    ]


def check_support(reporter: Reporter, data: Path) -> None:
    print("\n== frozen project-support products ==")
    for label, path in support_files(data):
        reporter.file(path, label)


def unique_manifest_stores(manifest: Path) -> set[str]:
    opener = gzip.open if manifest.suffix == ".gz" else open
    with opener(manifest, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        if "store" not in (reader.fieldnames or []):
            raise RuntimeError(f"manifest has no 'store' column: {manifest}")
        return {row["store"] for row in reader if row.get("store")}


def verify_manifest_stores(reporter: Reporter, manifest: Path, label: str) -> None:
    try:
        stores = unique_manifest_stores(manifest)
    except Exception as exc:
        reporter.add(FAIL, f"{label} manifest store scan", str(exc))
        return
    missing = [p for p in stores if not Path(p).is_file()]
    if missing:
        reporter.add(FAIL, f"{label} stores", f"{len(stores)-len(missing):,}/{len(stores):,} present; {len(missing):,} missing")
        for p in missing[:5]:
            print(f"       missing: {p}")
        if len(missing) > 5:
            print(f"       ... plus {len(missing)-5:,} more")
    else:
        reporter.add(PASS, f"{label} stores", f"{len(stores):,}/{len(stores):,} present")


def check_training(reporter: Reporter, data: Path, *, verify_stores: bool) -> None:
    print("\n== training inputs ==")
    for label, path in support_files(data)[:3]:
        reporter.file(path, label)

    native = data / "forecasts/hrrr/qpf6_native/metadata/hrrr_qpf6_manifest.csv.gz"
    if reporter.file(native, "HRRR native manifest") and verify_stores:
        verify_manifest_stores(reporter, native, "HRRR native")
    elif not verify_stores:
        reporter.add(WARN, "HRRR native daily stores", "not scanned; add --verify-stores for the complete manifest-vs-disk check")


def check_forecast_hour(reporter: Reporter, data: Path) -> None:
    print("\n== forecast-hour diagnostic archive ==")
    root = data / "derived/evaluation/stageiv_forecast_hour_v1"
    required = [
        root / "cases.csv",
        root / "valid_times.csv",
        root / "hrrr_qpf6_mm.npy",
        root / "stageiv_budget_qpf6_mm.npy",
        root / "preflight_report.json",
    ]
    for path in required:
        reporter.file(path, path.name)
    report = root / "preflight_report.json"
    if report.is_file():
        try:
            status = json.loads(report.read_text()).get("status")
            reporter.add(PASS if status == "PASS" else FAIL, "forecast-hour preflight status", str(status))
        except Exception as exc:
            reporter.add(FAIL, "forecast-hour preflight report", str(exc))


def check_diagnostics(reporter: Reporter, data: Path) -> None:
    check_support(reporter, data)
    check_forecast_hour(reporter, data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        nargs="?",
        default="all",
        choices=("setup", "support", "training", "diagnostics", "all"),
        help="subset of the reproducibility stack to validate",
    )
    parser.add_argument(
        "--verify-stores",
        action="store_true",
        help="scan every unique HRRR store path referenced by the training manifest",
    )
    args = parser.parse_args()

    reporter = Reporter()
    print("Contrastive-learning reproducibility preflight")
    print(f"Python: {sys.executable}")

    project, data = check_setup(reporter)
    if args.scope != "setup" and data is None:
        print("\nCannot continue without DATA_DIR.")
    elif data is not None:
        if args.scope == "support":
            check_support(reporter, data)
        elif args.scope == "training":
            check_training(reporter, data, verify_stores=args.verify_stores)
        elif args.scope == "diagnostics":
            check_diagnostics(reporter, data)
        elif args.scope == "all":
            check_training(reporter, data, verify_stores=args.verify_stores)
            check_diagnostics(reporter, data)

    print("\n== summary ==")
    print(f"Failures: {reporter.failed}")
    print(f"Warnings: {reporter.warned}")
    if reporter.failed:
        print("\nPreflight FAILED. See docs/DATA.md and docs/REPRODUCING_RESULTS.md for recovery/setup commands.")
        return 1
    print("Preflight PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
