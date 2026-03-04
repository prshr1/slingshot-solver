"""
Styled pandas DataFrame display utilities for Jupyter notebooks.

Provides conditional-formatting helpers that produce publication-quality
styled tables for candidate summaries, tier breakdowns, cross-run
comparisons, and uncertainty results.

Usage in a notebook cell::

    from slingshot.output.notebook_display import (
        styled_candidates_table,
        styled_tier_summary,
        styled_cross_run_table,
    )
    styled_candidates_table(analyses_best, top_indices)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    from pandas.io.formats.style import Styler
except (ImportError, Exception):  # pragma: no cover
    Styler = None  # type: ignore[assignment,misc]


# ───────────────────────────────────────────────────────────────────
# Colour palette
# ───────────────────────────────────────────────────────────────────
_TIER_COLORS = {
    "planet-dominated": "rgba(200, 230, 201, 0.35)",   # green-100, low opacity
    "hybrid":           "rgba(255, 224, 178, 0.35)",   # orange-100, low opacity
    "star-dominated":   "rgba(187, 222, 251, 0.35)",   # blue-100, low opacity
}

_GRADIENT_CMAP = "YlOrRd"  # warm gradient for Δv columns


# ───────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────

def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for notebook_display. Install with: pip install pandas")


def _tier_color_map(val: str) -> str:
    """Return background-color CSS for a tier label cell."""
    bg = _TIER_COLORS.get(str(val), "")
    return f"background-color: {bg}" if bg else ""


def _enc_get(enc: Any, key: str, default: Any = None) -> Any:
    if enc is None:
        return default
    if isinstance(enc, dict):
        return enc.get(key, default)
    return getattr(enc, key, default)


# ───────────────────────────────────────────────────────────────────
# Candidate table → styled DataFrame
# ───────────────────────────────────────────────────────────────────

def candidates_dataframe(
    analyses_best: Sequence[Optional[Dict[str, Any]]],
    top_indices: Optional[Sequence[int]] = None,
    top_n: Optional[int] = None,
) -> "pd.DataFrame":
    """Build a tidy DataFrame of re-run candidate metrics.

    Parameters
    ----------
    analyses_best : list of dict or None
        Analysis dicts from phase_rerun.
    top_indices : list of int, optional
        MC indices for each analysis entry.
    top_n : int, optional
        Limit to top-N by scalar Δv. If None, include all valid.

    Returns
    -------
    pd.DataFrame
    """
    _require_pandas()

    rows = []
    valid = [
        (i, a) for i, a in enumerate(analyses_best) if a is not None
    ]
    valid.sort(key=lambda x: float(x[1].get("delta_v", -np.inf)), reverse=True)
    if top_n is not None:
        valid = valid[:top_n]

    for rank, (i, ana) in enumerate(valid, start=1):
        mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
        enc = ana.get("encounter")
        r_star = _enc_get(enc, "r_star_min", np.nan)
        rows.append({
            "Rank": rank,
            "MC#": mc_idx,
            "Δv (km/s)": round(ana.get("delta_v", np.nan), 3),
            "Δv (%)": round(ana.get("delta_v_pct", np.nan), 2),
            "|ΔV_vec| (km/s)": round(ana.get("delta_v_vec", np.nan), 3),
            "½|ΔV|² (km²/s²)": round(ana.get("energy_half_dv_vec_sq", np.nan), 3),
            "Deflection (°)": round(ana.get("deflection", np.nan), 1),
            "r_min planet (km)": round(ana.get("r_min", np.nan), 0),
            "r_min star (km)": round(float(r_star) if np.isfinite(float(r_star)) else np.nan, 0),
            "Tier": ana.get("tier", ""),
            "ε/Δε ratio": round(ana["tier_ratio"], 3) if ana.get("tier_ratio") is not None else np.nan,
        })

    return pd.DataFrame(rows)


def styled_candidates_table(
    analyses_best: Sequence[Optional[Dict[str, Any]]],
    top_indices: Optional[Sequence[int]] = None,
    top_n: int = 30,
) -> "Styler":
    """Return a styled DataFrame for display in Jupyter.

    - Δv columns use a warm colour gradient.
    - Tier column cells are coloured by tier.
    - Numeric columns are right-aligned.

    Parameters
    ----------
    analyses_best : list
        From ``phase_rerun``.
    top_indices : list of int, optional
        MC indices.
    top_n : int
        Max rows to show (default 30).

    Returns
    -------
    pandas Styler
    """
    _require_pandas()
    df = candidates_dataframe(analyses_best, top_indices, top_n=top_n)
    if df.empty:
        return df.style

    styler = (
        df.style
        .hide(axis="index")
        .format(precision=3, na_rep="—")
        .map(_tier_color_map, subset=["Tier"])
        .bar(subset=["Δv (km/s)"], color="rgba(255, 204, 188, 0.3)", vmin=0)
        .bar(subset=["|ΔV_vec| (km/s)"], color="rgba(197, 202, 233, 0.3)", vmin=0)
        .set_properties(**{"text-align": "right"}, subset=df.select_dtypes("number").columns.tolist())
        .set_properties(**{"text-align": "center"}, subset=["MC#", "Tier"])
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "caption", "props": [("font-size", "1.1em"), ("font-weight", "bold")]},
        ])
        .set_caption(f"Top {len(df)} Candidates (sorted by scalar Δv)")
    )
    return styler


# ───────────────────────────────────────────────────────────────────
# Tier summary → styled DataFrame
# ───────────────────────────────────────────────────────────────────

def styled_tier_summary(
    tier_stats: Dict[str, Dict[str, Any]],
) -> "Styler":
    """Compact tier-summary table.

    Parameters
    ----------
    tier_stats : dict
        Output of ``tiering.tier_summary_stats()``.

    Returns
    -------
    pandas Styler
    """
    _require_pandas()
    from ..analysis.tiering import TIER_ORDER

    rows = []
    for tier in TIER_ORDER:
        ts = tier_stats.get(tier, {})
        rows.append({
            "Tier": tier,
            "Count": ts.get("count", 0),
            "Δv min": ts.get("dv_min"),
            "Δv median": ts.get("dv_median"),
            "Δv max": ts.get("dv_max"),
            "Ratio range": (
                f"{ts['ratio_min']:.2f}–{ts['ratio_max']:.2f}"
                if ts.get("ratio_min") is not None else "—"
            ),
            "Top MC#": ts.get("top_mc_idx"),
        })

    df = pd.DataFrame(rows)
    styler = (
        df.style
        .hide(axis="index")
        .format(precision=3, na_rep="—")
        .map(_tier_color_map, subset=["Tier"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
        ])
        .set_caption("Tier Classification Summary")
    )
    return styler


# ───────────────────────────────────────────────────────────────────
# Cross-run comparison → styled DataFrame
# ───────────────────────────────────────────────────────────────────

def styled_cross_run_table(
    comparison_df: "pd.DataFrame",
) -> "Styler":
    """Style a cross-run comparison DataFrame.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``compare_runs.build_comparison_df()``.

    Returns
    -------
    pandas Styler
    """
    _require_pandas()
    if comparison_df.empty:
        return comparison_df.style

    num_cols = comparison_df.select_dtypes("number").columns.tolist()
    styler = (
        comparison_df.style
        .hide(axis="index")
        .format(precision=3, na_rep="—")
        .set_properties(**{"text-align": "right"}, subset=num_cols)
        .set_properties(**{"text-align": "left"}, subset=["run"])
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
        ])
        .set_caption("Cross-Run Comparison")
    )

    # Highlight best Δv across runs
    dv_col = "best_dv_kms"
    if dv_col in comparison_df.columns:
        styler = styler.highlight_max(subset=[dv_col], color="#c8e6c9")

    return styler


# ───────────────────────────────────────────────────────────────────
# Uncertainty confidence intervals → styled DataFrame
# ───────────────────────────────────────────────────────────────────

def styled_uncertainty_table(
    ci_df: "pd.DataFrame",
    tier_lookup: Optional[Dict[int, str]] = None,
) -> "Styler":
    """Style the confidence-interval DataFrame from uncertainty analysis.

    Parameters
    ----------
    ci_df : pd.DataFrame
        From ``uncertainty.compute_confidence_bands()``.
    tier_lookup : dict, optional
        ``{mc_idx: tier_label}`` for colouring.

    Returns
    -------
    pandas Styler
    """
    _require_pandas()
    if ci_df.empty:
        return ci_df.style

    display_df = ci_df.copy()
    if tier_lookup and "candidate_mc_idx" in display_df.columns:
        display_df.insert(
            1, "Tier",
            display_df["candidate_mc_idx"].map(
                lambda x: tier_lookup.get(int(x), "")
            ),
        )

    styler = (
        display_df.style
        .hide(axis="index")
        .format(precision=3, na_rep="—")
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
        ])
        .set_caption("Parameter Posterior Confidence Intervals")
    )

    if "Tier" in display_df.columns:
        styler = styler.map(_tier_color_map, subset=["Tier"])

    return styler
