"""
Uncertainty propagation and Pareto-front stability analysis.

Provides three main capabilities:

1. **Parameter-posterior Monte Carlo** — re-run top candidates with system
   parameters drawn from measurement posteriors to produce confidence bands
   on all metrics.

2. **Confidence bands** — percentile-based CI extraction from posterior draws.

3. **Bootstrap Pareto stability** — resample the existing MC population and
   re-run Pareto selection to measure how often each candidate appears on
   the front (no re-integration required).
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import FullConfig
from ..constants import M_SUN, M_JUP, R_JUP, R_SUN, AU_KM
from ..core.dynamics import init_hot_jupiter_barycentric, simulate_3body
from .trajectory import analyze_trajectory
from .monte_carlo import (
    select_pareto_indices,
    select_weighted_indices,
    select_top_indices,
    _pareto_front_mask,
    _prepare_objective_matrix,
)


# ───────────────────────────────────────────────────────────────────
# 1. Parameter-posterior Monte Carlo
# ───────────────────────────────────────────────────────────────────

def _draw_system_params(
    base_cfg: FullConfig,
    rng: np.random.Generator,
    param_dists: Dict[str, Dict[str, float]],
) -> FullConfig:
    """Return a *copy* of ``base_cfg`` with system params drawn from posteriors.

    Parameters
    ----------
    base_cfg : FullConfig
        Reference configuration (not mutated).
    rng : Generator
        Numpy random generator.
    param_dists : dict
        ``{field_name: {"mean": float, "std": float}}``.
        Field names correspond to ``SystemConfig`` attributes, e.g.
        ``"M_planet_Mjup"``, ``"a_planet_AU"``, ``"M_star_Msun"``.
    """
    cfg = base_cfg.model_copy(deep=True)
    sys_dict = cfg.system.model_dump()

    for field, dist in param_dists.items():
        if field not in sys_dict:
            raise KeyError(f"Unknown SystemConfig field: '{field}'")
        mean = dist["mean"]
        std = dist["std"]
        drawn = rng.normal(mean, std)
        # Clamp to positive for physical quantities
        drawn = max(drawn, 1e-6)
        setattr(cfg.system, field, drawn)

    return cfg


def run_parameter_posterior_mc(
    base_cfg: FullConfig,
    mc_results: Dict[str, Any],
    top_indices: np.ndarray,
    n_draws: int = 50,
    seed: Optional[int] = None,
    param_dists: Optional[Dict[str, Dict[str, float]]] = None,
    metric_keys: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run top candidates under parameter-posterior draws.

    For each draw, system parameters (masses, radii, semi-major axis) are
    sampled from Gaussian posteriors.  The star–planet initial state is
    rebuilt and each selected candidate is re-integrated and re-analysed.

    Parameters
    ----------
    base_cfg : FullConfig
        Reference configuration.
    mc_results : dict
        Results dict from ``run_monte_carlo`` (needs ``Y_sp0``, ``sat_states``).
    top_indices : array-like
        MC indices of candidates to re-run (from Pareto / selection).
    n_draws : int
        Number of posterior draws.
    seed : int, optional
        Random seed for posterior sampling.
    param_dists : dict, optional
        ``{field: {"mean": ..., "std": ...}}``.  If None, uses Kepler-432
        defaults.
    metric_keys : list of str, optional
        Metric names to collect.  Defaults to the standard set.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: ``draw``, ``candidate_mc_idx``, plus one column per metric.
    """
    if param_dists is None:
        param_dists = {
            "M_planet_Mjup": {"mean": base_cfg.system.M_planet_Mjup, "std": 0.4},
            "a_planet_AU": {"mean": base_cfg.system.a_planet_AU, "std": 0.002},
            "M_star_Msun": {"mean": base_cfg.system.M_star_Msun, "std": 0.05},
        }

    if metric_keys is None:
        metric_keys = [
            "delta_v", "delta_v_vec", "energy_half_dv_vec_sq",
            "deflection", "delta_v_planet_frame", "energy_from_planet_orbit",
        ]

    rng = np.random.default_rng(seed)
    sat_states = mc_results["sat_states"]

    rows: List[Dict[str, Any]] = []

    for d in range(n_draws):
        if verbose and d % max(1, n_draws // 10) == 0:
            print(f"  Uncertainty draw {d + 1}/{n_draws}")

        cfg_draw = _draw_system_params(base_cfg, rng, param_dists)
        m_star = cfg_draw.system.M_star_Msun * M_SUN
        m_p = cfg_draw.system.M_planet_Mjup * M_JUP
        R_p = cfg_draw.system.R_planet_Rjup * R_JUP

        # Rebuild star–planet initial state with drawn parameters
        Y_sp0_draw = init_hot_jupiter_barycentric(
            m_star=m_star, m_p=m_p,
            bulk_velocity_vx_kms=cfg_draw.system.bulk_velocity_vx_kms,
            bulk_velocity_vy_kms=cfg_draw.system.bulk_velocity_vy_kms,
        )

        for idx in top_indices:
            x3, y3, vx3, vy3 = sat_states[idx]
            xs, ys, vxs, vys, xp, yp, vxp, vyp = Y_sp0_draw
            Y0 = np.array([xs, ys, vxs, vys, xp, yp, vxp, vyp,
                            x3, y3, vx3, vy3], dtype=float)

            r0_bary = np.hypot(x3, y3)
            escape_r = r0_bary * cfg_draw.numerical.escape_radius_factor

            try:
                sol = simulate_3body(
                    Y0,
                    (0.0, cfg_draw.pipeline.t_best_max_sec),
                    m_star=m_star, m_p=m_p,
                    n_eval=0,
                    rtol=cfg_draw.numerical.rtol,
                    atol=cfg_draw.numerical.atol,
                    escape_radius_km=escape_r,
                    method=cfg_draw.numerical.ode_method,
                    softening_km=cfg_draw.numerical.softening_km,
                )

                ana = analyze_trajectory(
                    sol, frame="barycentric",
                    m_star=m_star, m_p=m_p, R_p=R_p,
                    r_far_factor=cfg_draw.numerical.r_far_factor,
                    min_clearance_factor=cfg_draw.numerical.min_clearance_factor,
                )
            except Exception:
                ana = None

            row: Dict[str, Any] = {
                "draw": d,
                "candidate_mc_idx": int(idx),
            }
            # Record drawn system params
            for field in param_dists:
                row[f"drawn_{field}"] = getattr(cfg_draw.system, field)

            if ana is not None:
                for key in metric_keys:
                    row[key] = ana.get(key, np.nan)
            else:
                for key in metric_keys:
                    row[key] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────
# 2. Confidence bands
# ───────────────────────────────────────────────────────────────────

def compute_confidence_bands(
    posterior_df: pd.DataFrame,
    metric_names: Optional[List[str]] = None,
    ci_levels: Sequence[float] = (0.68, 0.95),
    group_by: str = "candidate_mc_idx",
) -> pd.DataFrame:
    """Compute percentile-based confidence intervals from posterior draws.

    Parameters
    ----------
    posterior_df : DataFrame
        Output of ``run_parameter_posterior_mc``.
    metric_names : list of str, optional
        Metrics to summarize.  Default: all numeric columns except
        ``draw`` and ``candidate_mc_idx``.
    ci_levels : sequence of float
        Confidence levels (e.g. 0.68, 0.95).
    group_by : str
        Column to group candidates by.

    Returns
    -------
    DataFrame
        One row per candidate, columns: metric_median, metric_lo68/hi68, etc.
    """
    if metric_names is None:
        exclude = {"draw", group_by}
        metric_names = [
            c for c in posterior_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(posterior_df[c])
            and not c.startswith("drawn_")
        ]

    records: List[Dict[str, Any]] = []

    for cand_idx, grp in posterior_df.groupby(group_by):
        rec: Dict[str, Any] = {group_by: cand_idx, "n_draws": len(grp)}

        for metric in metric_names:
            vals = grp[metric].dropna().values
            if len(vals) == 0:
                rec[f"{metric}_median"] = np.nan
                for ci in ci_levels:
                    tag = f"{int(ci * 100)}"
                    rec[f"{metric}_lo{tag}"] = np.nan
                    rec[f"{metric}_hi{tag}"] = np.nan
                continue

            rec[f"{metric}_median"] = float(np.median(vals))
            rec[f"{metric}_mean"] = float(np.mean(vals))
            rec[f"{metric}_std"] = float(np.std(vals))
            for ci in ci_levels:
                lo_pct = (1.0 - ci) / 2.0 * 100.0
                hi_pct = (1.0 + ci) / 2.0 * 100.0
                tag = f"{int(ci * 100)}"
                rec[f"{metric}_lo{tag}"] = float(np.percentile(vals, lo_pct))
                rec[f"{metric}_hi{tag}"] = float(np.percentile(vals, hi_pct))

        records.append(rec)

    return pd.DataFrame(records)


# ───────────────────────────────────────────────────────────────────
# 3. Bootstrap Pareto stability
# ───────────────────────────────────────────────────────────────────

def bootstrap_pareto_stability(
    mc: Dict[str, Any],
    objectives: Sequence[Dict[str, Any]],
    n_resample: int = 200,
    seed: Optional[int] = None,
    top_frac: float = 0.10,
    min_top: int = 5,
    tie_break_normalization: str = "minmax",
) -> Dict[str, Any]:
    """Measure Pareto-front membership stability via bootstrap resampling.

    For each bootstrap iteration, we resample the successful MC population
    *with replacement*, re-run Pareto selection, and record which original
    MC indices appear on the front.

    Parameters
    ----------
    mc : dict
        Full MC results dict (needs ``ok`` and metric arrays).
    objectives : sequence of dict
        Objective descriptors (same format as ``select_pareto_indices``).
    n_resample : int
        Number of bootstrap iterations.
    seed : int, optional
        Random seed.
    top_frac, min_top, tie_break_normalization
        Pareto selection parameters (mirror ``select_pareto_indices``).

    Returns
    -------
    dict
        ``membership_freq``  — dict mapping MC index → fraction of bootstraps
            it appeared in the selected set.
        ``stable_front``     — list of MC indices appearing in ≥50% of bootstraps.
        ``n_resample``       — actual number of bootstraps performed.
    """
    rng = np.random.default_rng(seed)
    ok = np.asarray(mc["ok"])
    ok_indices = np.where(ok)[0]
    n_ok = ok_indices.size

    if n_ok == 0:
        return {"membership_freq": {}, "stable_front": [], "n_resample": n_resample}

    # Build objective matrix once
    ok_idx_full, transformed_full, weights = _prepare_objective_matrix(mc, objectives)

    # Map from ok_idx_full position → original MC index
    full_pos_to_mc = {int(ok_idx_full[i]): i for i in range(len(ok_idx_full))}

    counts: Dict[int, int] = {}

    for _ in range(n_resample):
        # Resample indices (with replacement) from the ok population
        boot_local = rng.choice(n_ok, size=n_ok, replace=True)
        boot_mc_indices = ok_indices[boot_local]

        # Build a synthetic MC dict with resampled population
        mc_boot: Dict[str, Any] = {"ok": np.ones(n_ok, dtype=bool)}
        for key, val in mc.items():
            if key == "ok":
                continue
            if isinstance(val, np.ndarray) and val.shape[0] == ok.shape[0]:
                mc_boot[key] = val[boot_mc_indices]

        # Run Pareto selection on resampled population
        selected_boot = select_pareto_indices(
            mc_boot,
            objectives=objectives,
            top_frac=top_frac,
            min_top=min_top,
            tie_break_normalization=tie_break_normalization,
        )

        # Map selected back to original MC indices (deduplicate per resample)
        seen_this_boot: set = set()
        for sel_local in selected_boot:
            orig_idx = int(boot_mc_indices[sel_local])
            if orig_idx not in seen_this_boot:
                seen_this_boot.add(orig_idx)
                counts[orig_idx] = counts.get(orig_idx, 0) + 1

    freq = {idx: cnt / n_resample for idx, cnt in counts.items()}
    stable = sorted(idx for idx, f in freq.items() if f >= 0.5)

    return {
        "membership_freq": freq,
        "stable_front": stable,
        "n_resample": n_resample,
    }
