"""
Cross-run comparison utilities.

Load ``summary.csv`` and ``config.yaml`` from multiple result directories
and produce a side-by-side comparison table, with optional tier-breakdown
and auto-discovery of result directories.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..console import safe_print as print

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ───────────────────────────────────────────────────────────────────
# Auto-discovery
# ───────────────────────────────────────────────────────────────────

def discover_runs(
    results_root: str = "results",
    system_filter: Optional[str] = None,
) -> List[str]:
    """Find all ``results_*`` directories under *results_root*.

    Parameters
    ----------
    results_root : str
        Parent folder containing result directories.
    system_filter : str, optional
        If given, only return runs whose folder name contains this string
        (case-insensitive).

    Returns
    -------
    list of str
        Sorted list of absolute directory paths.
    """
    root = Path(results_root)
    if not root.is_dir():
        return []
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("results_")
    )
    if system_filter:
        filt = system_filter.lower()
        dirs = [d for d in dirs if filt in d.name.lower()]
    return [str(d) for d in dirs]


# ───────────────────────────────────────────────────────────────────
# Single-run loader
# ───────────────────────────────────────────────────────────────────

def _load_run_summary(run_dir: Path) -> Dict[str, Any]:
    """Load config and summary from a single run directory."""
    info: Dict[str, Any] = {"dir": str(run_dir)}

    cfg_path = run_dir / "config.yaml"
    config: Dict[str, Any] = {}
    if cfg_path.exists() and yaml is not None:
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    info["config"] = config

    sys_cfg = config.get("system", {})
    info["system"] = sys_cfg.get("name", "?")
    pipe_cfg = config.get("pipeline", {})
    info["N"] = pipe_cfg.get("N_particles", "?")
    info["select_mode"] = pipe_cfg.get("select_mode", "single")

    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

        info["n_candidates"] = len(rows)

        # Metric extraction
        try:
            dvs = []
            for row in rows:
                raw = row.get("dv_kms")
                if raw is None:
                    raw = row.get("\u0394v (km/s)", row.get("dv (km/s)", 0))
                dvs.append(float(raw))
            info["best_dv"] = max(dvs) if dvs else None
            info["median_dv"] = float(sorted(dvs)[len(dvs) // 2]) if dvs else None
        except (ValueError, TypeError):
            info["best_dv"] = None
            info["median_dv"] = None

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

        # Tier breakdown (if available in CSV)
        tier_counts: Dict[str, int] = {}
        for row in rows:
            t = row.get("tier", "")
            if t:
                tier_counts[t] = tier_counts.get(t, 0) + 1
        info["tier_counts"] = tier_counts if tier_counts else None
    else:
        info["n_candidates"] = 0
        info["best_dv"] = None
        info["median_dv"] = None
        info["best_dv_vec"] = None
        info["best_half_dv_sq"] = None
        info["tier_counts"] = None

    return info


# ───────────────────────────────────────────────────────────────────
# Comparison builder (DataFrame)
# ───────────────────────────────────────────────────────────────────

def build_comparison_df(run_dirs: List[str]) -> "pd.DataFrame":
    """Build a tidy pandas DataFrame comparing multiple runs.

    Parameters
    ----------
    run_dirs : list of str
        Paths to result directories.

    Returns
    -------
    pd.DataFrame
        One row per run with key metrics.

    Raises
    ------
    ImportError
        If pandas is not available.
    """
    if pd is None:
        raise ImportError("pandas is required for build_comparison_df")

    summaries = [_load_run_summary(Path(d)) for d in run_dirs if Path(d).is_dir()]

    records = []
    for s in summaries:
        row: Dict[str, Any] = {
            "run": Path(s["dir"]).name,
            "system": s["system"],
            "select_mode": s.get("select_mode", "single"),
            "N_particles": s["N"],
            "n_candidates": s["n_candidates"],
            "best_dv_kms": s["best_dv"],
            "median_dv_kms": s.get("median_dv"),
            "best_dv_vec_kms": s["best_dv_vec"],
            "best_half_dv_sq": s["best_half_dv_sq"],
        }
        tc = s.get("tier_counts")
        if tc:
            row["n_planet"] = tc.get("planet-dominated", 0)
            row["n_hybrid"] = tc.get("hybrid", 0)
            row["n_star"] = tc.get("star-dominated", 0)
        records.append(row)

    return pd.DataFrame(records)


# ───────────────────────────────────────────────────────────────────
# Text-table comparison (legacy)
# ───────────────────────────────────────────────────────────────────

def compare_runs(run_dirs: List[str]) -> Dict[str, Any]:
    """Compare multiple pipeline run directories.

    Returns
    -------
    dict
        ``summaries``, ``table`` (text), and ``df`` (DataFrame if pandas available).
    """
    summaries = []
    for run_dir in run_dirs:
        p = Path(run_dir)
        if p.is_dir():
            summaries.append(_load_run_summary(p))

    header = (
        f"{'Run':50s} | {'System':15s} | {'Mode':8s} | {'N':>6s} | {'Cands':>5s} | "
        f"{'dV':>8s} | {'|dV_vec|':>8s} | {'0.5|dV|^2':>10s}"
    )
    sep = "-" * len(header)
    rows_txt = [header, sep]
    for s in summaries:
        best_dv = f"{s['best_dv']:.2f}" if s["best_dv"] is not None else "-"
        best_vec = f"{s['best_dv_vec']:.2f}" if s["best_dv_vec"] is not None else "-"
        best_half = f"{s['best_half_dv_sq']:.2f}" if s["best_half_dv_sq"] is not None else "-"
        rows_txt.append(
            f"{Path(s['dir']).name:50s} | {s['system']:15s} | {s.get('select_mode', 'single'):8s} | {str(s['N']):>6s} | "
            f"{s['n_candidates']:>5d} | {best_dv:>8s} | {best_vec:>8s} | {best_half:>10s}"
        )

    table = "\n".join(rows_txt)
    result: Dict[str, Any] = {"summaries": summaries, "table": table}

    if pd is not None:
        result["df"] = build_comparison_df(run_dirs)

    return result


def print_comparison(run_dirs: List[str]) -> None:
    """Compare runs and print the table."""
    result = compare_runs(run_dirs)
    print(result["table"])
