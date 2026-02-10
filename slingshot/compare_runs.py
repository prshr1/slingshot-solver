"""
Cross-run comparison utilities.

Load ``summary.csv`` and ``config.yaml`` from multiple result directories
and produce a side-by-side comparison table + optional overlay plots.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None


def _load_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Load config and summary from a single run directory.

    Returns a dict with keys: ``dir``, ``system``, ``N``, ``best_dv``,
    ``best_dv_vec``, ``best_half_dv_sq``, ``n_candidates``, ``config``.
    """
    info: Dict[str, Any] = {"dir": str(run_dir)}

    # Config
    cfg_path = run_dir / "config.yaml"
    config = {}
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path) as f:
            config = yaml.safe_load(f) or {}
    info["config"] = config
    sys_cfg = config.get("system", {})
    info["system"] = sys_cfg.get("name", "?")
    info["N"] = config.get("pipeline", {}).get("N_particles", "?")

    # Summary CSV
    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        info["n_candidates"] = len(rows)
        # Best metrics
        try:
            dvs = [float(r.get("dv_kms", r.get("Δv (km/s)", 0))) for r in rows]
            info["best_dv"] = max(dvs) if dvs else None
        except (ValueError, TypeError):
            info["best_dv"] = None
        try:
            dv_vecs = [float(r.get("dv_vec_kms", 0)) for r in rows]
            info["best_dv_vec"] = max(dv_vecs) if dv_vecs else None
        except (ValueError, TypeError):
            info["best_dv_vec"] = None
        try:
            halfs = [float(r.get("half_dv_vec_sq", 0)) for r in rows]
            info["best_half_dv_sq"] = max(halfs) if halfs else None
        except (ValueError, TypeError):
            info["best_half_dv_sq"] = None
    else:
        info["n_candidates"] = 0
        info["best_dv"] = None
        info["best_dv_vec"] = None
        info["best_half_dv_sq"] = None

    return info


def compare_runs(run_dirs: List[str]) -> Dict[str, Any]:
    """Compare multiple pipeline run directories.

    Parameters
    ----------
    run_dirs : list of str
        Paths to result directories (each containing config.yaml + summary.csv).

    Returns
    -------
    dict
        ``summaries``: list of per-run dicts.
        ``table``: formatted comparison string.
        ``df``: pandas DataFrame (if pandas available).
    """
    summaries = []
    for d in run_dirs:
        p = Path(d)
        if p.is_dir():
            summaries.append(_load_run_summary(p))

    # Build text table
    header = f"{'Run':50s} | {'System':15s} | {'N':>6s} | {'Cands':>5s} | {'Δv':>8s} | {'|ΔV_vec|':>8s} | {'½|ΔV|²':>8s}"
    sep = "-" * len(header)
    rows_txt = [header, sep]
    for s in summaries:
        best_dv = f"{s['best_dv']:.2f}" if s['best_dv'] is not None else "—"
        best_vec = f"{s['best_dv_vec']:.2f}" if s['best_dv_vec'] is not None else "—"
        best_half = f"{s['best_half_dv_sq']:.2f}" if s['best_half_dv_sq'] is not None else "—"
        rows_txt.append(
            f"{Path(s['dir']).name:50s} | {s['system']:15s} | {str(s['N']):>6s} | "
            f"{s['n_candidates']:>5d} | {best_dv:>8s} | {best_vec:>8s} | {best_half:>8s}"
        )
    table = "\n".join(rows_txt)

    result: Dict[str, Any] = {
        "summaries": summaries,
        "table": table,
    }

    if pd is not None:
        df_rows = []
        for s in summaries:
            df_rows.append({
                "run": Path(s["dir"]).name,
                "system": s["system"],
                "N_particles": s["N"],
                "n_candidates": s["n_candidates"],
                "best_dv_kms": s["best_dv"],
                "best_dv_vec_kms": s["best_dv_vec"],
                "best_half_dv_sq": s["best_half_dv_sq"],
            })
        result["df"] = pd.DataFrame(df_rows)

    return result


def print_comparison(run_dirs: List[str]) -> None:
    """Compare runs and print the table."""
    result = compare_runs(run_dirs)
    print(result["table"])
