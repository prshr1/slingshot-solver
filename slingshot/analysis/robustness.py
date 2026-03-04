"""
Numerical robustness and sensitivity analysis.

Provides two main capabilities:

1. **N-convergence test** — run the MC pipeline at increasing particle counts
   with a fixed seed and measure when key metrics stabilise.

2. **Numerical sensitivity sweep** — vary one numerical parameter at a time
   (softening, tolerance, star clearance, flyby filter, sampling bounds)
   and record the effect on best-candidate metrics.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import FullConfig, RobustnessConfig
from ..constants import M_SUN, M_JUP, R_JUP, R_SUN
from .monte_carlo import run_monte_carlo, select_top_indices, resolve_metric_array


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def _derive_physics(cfg: FullConfig):
    m_star = cfg.system.M_star_Msun * M_SUN
    m_p = cfg.system.M_planet_Mjup * M_JUP
    R_p = cfg.system.R_planet_Rjup * R_JUP
    R_star = cfg.system.R_star_Rsun * R_SUN
    return m_star, m_p, R_p, R_star


def _run_mc_and_extract_metrics(
    cfg: FullConfig,
    metric_names: List[str],
    seed: int,
    label: str = "",
    verbose: bool = False,
) -> Dict[str, float]:
    """Run a single MC sweep and return best-candidate metric values."""
    m_star, m_p, R_p, R_star = _derive_physics(cfg)
    rng = np.random.default_rng(seed)

    mc = run_monte_carlo(
        N=cfg.pipeline.N_particles,
        t_span=(0.0, cfg.pipeline.t_mc_max_sec),
        m_star=m_star, m_p=m_p, R_p=R_p,
        frame="barycentric",
        sampling_mode=cfg.sampling.mode,
        n_parallel=cfg.pipeline.n_parallel,
        rng=rng,
        verbose=False,
        v_mag_min=cfg.sampling.v_mag_min_kms,
        v_mag_max=cfg.sampling.v_mag_max_kms,
        impact_param_min_AU=cfg.sampling.impact_param_min_AU,
        impact_param_max_AU=cfg.sampling.impact_param_max_AU,
        angle_in_min_deg=cfg.sampling.angle_in_min_deg,
        angle_in_max_deg=cfg.sampling.angle_in_max_deg,
        r_init_AU=cfg.sampling.r_init_AU,
        bulk_velocity_vx_kms=cfg.system.bulk_velocity_vx_kms,
        bulk_velocity_vy_kms=cfg.system.bulk_velocity_vy_kms,
        rtol=cfg.numerical.rtol,
        atol=cfg.numerical.atol,
        r_far_factor=cfg.numerical.r_far_factor,
        min_clearance_factor=cfg.numerical.min_clearance_factor,
        bary_unbound_requirement=cfg.sampling.bary_unbound_requirement,
        flyby_r_min_max_hill=cfg.numerical.flyby_r_min_max_hill,
        escape_radius_factor=cfg.numerical.escape_radius_factor,
        ode_method=cfg.numerical.ode_method,
        softening_km=cfg.numerical.softening_km,
        star_min_clearance_Rstar=cfg.numerical.star_min_clearance_Rstar,
        R_star_Rsun=cfg.system.R_star_Rsun,
    )

    ok = mc["ok"]
    n_ok = int(np.sum(ok))
    result: Dict[str, float] = {"n_ok": float(n_ok)}

    if n_ok == 0:
        for m in metric_names:
            result[m] = np.nan
        return result

    ok_idx = np.where(ok)[0]
    for m in metric_names:
        try:
            arr = resolve_metric_array(mc, m)
            vals = arr[ok_idx]
            finite = vals[np.isfinite(vals)]
            result[m] = float(np.max(finite)) if finite.size > 0 else np.nan
        except KeyError:
            result[m] = np.nan

    return result


# ───────────────────────────────────────────────────────────────────
# 1. N-convergence test
# ───────────────────────────────────────────────────────────────────

def run_convergence_test(
    base_cfg: FullConfig,
    N_values: Optional[List[int]] = None,
    seed: int = 42,
    metric_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run MC at increasing N and track metric convergence.

    Parameters
    ----------
    base_cfg : FullConfig
        Reference configuration (N_particles will be overridden per step).
    N_values : list of int, optional
        Particle counts to test.  Default: ``[500, 1000, 2000, 4000, 8000]``.
    seed : int
        Fixed seed for all runs.
    metric_names : list of str, optional
        Metrics to track.  Default: ``["delta_v", "delta_v_vec", "energy_from_planet_orbit"]``.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per N value, columns: ``N``, ``n_ok``, and one per metric.
        An extra ``converged_<metric>`` boolean column is added using the
        <1% relative change criterion against the previous doubling.
    """
    if N_values is None:
        N_values = [500, 1000, 2000, 4000, 8000]
    if metric_names is None:
        metric_names = ["delta_v", "delta_v_vec", "energy_from_planet_orbit"]

    rows: List[Dict[str, Any]] = []

    for i, N in enumerate(sorted(N_values)):
        if verbose:
            print(f"  Convergence test: N={N}")

        cfg = base_cfg.model_copy(deep=True)
        cfg.pipeline.N_particles = N

        result = _run_mc_and_extract_metrics(cfg, metric_names, seed=seed, verbose=False)
        result["N"] = N
        rows.append(result)

    df = pd.DataFrame(rows)

    # Compute convergence flags: <1% relative change from previous step
    for m in metric_names:
        col = f"converged_{m}"
        df[col] = False
        for i in range(1, len(df)):
            prev = df.loc[i - 1, m]
            curr = df.loc[i, m]
            if np.isfinite(prev) and np.isfinite(curr) and abs(prev) > 1e-12:
                rel_change = abs(curr - prev) / abs(prev)
                df.loc[i, col] = rel_change < 0.01

    return df


# ───────────────────────────────────────────────────────────────────
# 2. Numerical sensitivity sweep
# ───────────────────────────────────────────────────────────────────

def run_numerical_sensitivity(
    base_cfg: FullConfig,
    seed: int = 42,
    metric_names: Optional[List[str]] = None,
    robustness_cfg: Optional[RobustnessConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep one numerical parameter at a time and record metric changes.

    Parameters
    ----------
    base_cfg : FullConfig
        Reference configuration.
    seed : int
        Fixed seed for all runs.
    metric_names : list of str, optional
        Metrics to track.
    robustness_cfg : RobustnessConfig, optional
        Override sweep values.  If None uses ``base_cfg.robustness``.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: ``parameter``, ``value``, ``n_ok``, per-metric columns,
        and ``pct_change_<metric>`` relative to the reference run.
    """
    if metric_names is None:
        metric_names = ["delta_v", "delta_v_vec", "energy_from_planet_orbit"]

    if robustness_cfg is None:
        robustness_cfg = base_cfg.robustness

    # Reference run
    if verbose:
        print("  Sensitivity: reference run")
    ref = _run_mc_and_extract_metrics(base_cfg, metric_names, seed=seed)

    rows: List[Dict[str, Any]] = []
    ref_row = {"parameter": "reference", "value": "baseline", "n_ok": ref["n_ok"]}
    for m in metric_names:
        ref_row[m] = ref[m]
        ref_row[f"pct_change_{m}"] = 0.0
    rows.append(ref_row)

    def _sweep(param_name: str, values, setter):
        for val in values:
            if verbose:
                print(f"  Sensitivity: {param_name}={val}")
            cfg = base_cfg.model_copy(deep=True)
            setter(cfg, val)
            result = _run_mc_and_extract_metrics(cfg, metric_names, seed=seed)
            row: Dict[str, Any] = {
                "parameter": param_name,
                "value": val,
                "n_ok": result["n_ok"],
            }
            for m in metric_names:
                row[m] = result[m]
                if np.isfinite(ref[m]) and abs(ref[m]) > 1e-12:
                    row[f"pct_change_{m}"] = (result[m] - ref[m]) / abs(ref[m]) * 100.0
                else:
                    row[f"pct_change_{m}"] = np.nan
            rows.append(row)

    # Softening
    _sweep(
        "softening_km",
        robustness_cfg.softening_values,
        lambda cfg, v: setattr(cfg.numerical, "softening_km", v),
    )

    # Tolerance (apply to both rtol and atol)
    _sweep(
        "rtol_atol",
        robustness_cfg.tolerance_values,
        lambda cfg, v: (
            setattr(cfg.numerical, "rtol", v),
            setattr(cfg.numerical, "atol", v),
        ),
    )

    # Star clearance
    _sweep(
        "star_min_clearance_Rstar",
        robustness_cfg.clearance_values,
        lambda cfg, v: setattr(cfg.numerical, "star_min_clearance_Rstar", v),
    )

    # Flyby filter
    _sweep(
        "flyby_r_min_max_hill",
        robustness_cfg.flyby_values,
        lambda cfg, v: setattr(cfg.numerical, "flyby_r_min_max_hill", v),
    )

    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────
# 3. Formatting
# ───────────────────────────────────────────────────────────────────

def format_robustness_table(
    df: pd.DataFrame,
    metric_names: Optional[List[str]] = None,
    fmt: str = "markdown",
) -> str:
    """Produce a publication-ready table from sensitivity results.

    Parameters
    ----------
    df : DataFrame
        Output of ``run_numerical_sensitivity`` or ``run_convergence_test``.
    metric_names : list of str, optional
        Metrics to include (default: all numeric non-pct columns).
    fmt : str
        ``"markdown"`` or ``"latex"``.

    Returns
    -------
    str
        Formatted table string.
    """
    if metric_names is None:
        metric_names = [
            c for c in df.columns
            if c not in ("parameter", "value", "N", "n_ok")
            and not c.startswith("pct_change_")
            and not c.startswith("converged_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    if fmt == "latex":
        return _format_latex(df, metric_names)
    elif fmt == "markdown":
        return _format_markdown(df, metric_names)
    else:
        raise ValueError(f"Unknown format {fmt!r}, expected 'markdown' or 'latex'")


def _format_markdown(df: pd.DataFrame, metric_names: List[str]) -> str:
    lines: List[str] = []

    # Determine row label column
    if "parameter" in df.columns:
        label_col = "parameter"
        value_col = "value"
    elif "N" in df.columns:
        label_col = "N"
        value_col = None
    else:
        label_col = df.columns[0]
        value_col = None

    # Header
    cols = [label_col]
    if value_col:
        cols.append(value_col)
    cols.append("n_ok")
    cols.extend(metric_names)
    pct_cols = [f"pct_change_{m}" for m in metric_names if f"pct_change_{m}" in df.columns]
    cols.extend(pct_cols)
    conv_cols = [c for c in df.columns if c.startswith("converged_")]
    cols.extend(conv_cols)

    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines.append(header)
    lines.append(sep)

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float) and np.isfinite(v):
                if c.startswith("pct_change_"):
                    cells.append(f"{v:+.2f}%")
                else:
                    cells.append(f"{v:.4f}")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _format_latex(df: pd.DataFrame, metric_names: List[str]) -> str:
    lines: List[str] = []

    if "parameter" in df.columns:
        label_col = "parameter"
        value_col = "value"
    elif "N" in df.columns:
        label_col = "N"
        value_col = None
    else:
        label_col = df.columns[0]
        value_col = None

    cols = [label_col]
    if value_col:
        cols.append(value_col)
    cols.append("n_ok")
    cols.extend(metric_names)

    ncols = len(cols)
    lines.append(r"\begin{tabular}{" + "l" * ncols + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(str(c).replace("_", r"\_") for c in cols) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float) and np.isfinite(v):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)
