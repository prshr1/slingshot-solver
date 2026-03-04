"""
Physics-based candidate tiering engine.

Classifies gravitational slingshot candidates into three tiers based on the
ratio of planet-orbit energy extraction (ε_planet) to the monopole energy
change (Δε_monopole):

- **Planet-dominated** (ε_planet / |Δε_monopole| > threshold_high):
  The slingshot gain is driven primarily by the planet's orbital motion.
- **Hybrid** (threshold_low ≤ ratio ≤ threshold_high):
  Both star-only scattering and planet extraction contribute meaningfully.
- **Star-dominated** (ratio < threshold_low):
  The energy change is mostly from the star's gravitational potential
  (monopole scattering) with minimal three-body leverage.

Default thresholds: planet-dominated > 2.0, hybrid 0.5–2.0, star-dominated < 0.5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────────
# Tier labels (stable strings used as dict keys, CSV values, etc.)
# ───────────────────────────────────────────────────────────────────
TIER_PLANET = "planet-dominated"
TIER_HYBRID = "hybrid"
TIER_STAR = "star-dominated"

TIER_ORDER = [TIER_PLANET, TIER_HYBRID, TIER_STAR]
"""Canonical display order for tiers."""


# ───────────────────────────────────────────────────────────────────
# Single-candidate classification
# ───────────────────────────────────────────────────────────────────

def classify_candidate(
    analysis: Dict[str, Any],
    planet_threshold: float = 2.0,
    hybrid_threshold: float = 0.5,
) -> str:
    """Classify a single candidate into a physics-based tier.

    Parameters
    ----------
    analysis : dict
        Analysis dict produced by ``analyze_trajectory``.  Must contain
        ``energy_from_planet_orbit`` (ε_planet) and ``delta_eps_monopole``
        (Δε_monopole).
    planet_threshold : float
        Minimum ratio for the *planet-dominated* tier (default 2.0).
    hybrid_threshold : float
        Minimum ratio for the *hybrid* tier (default 0.5).

    Returns
    -------
    str
        One of ``TIER_PLANET``, ``TIER_HYBRID``, ``TIER_STAR``.
    """
    eps_planet = analysis.get("energy_from_planet_orbit")
    d_eps_mono = analysis.get("delta_eps_monopole")

    # Handle missing / non-finite values conservatively
    if eps_planet is None or d_eps_mono is None:
        return TIER_STAR
    eps_planet = float(eps_planet)
    d_eps_mono = float(d_eps_mono)
    if not np.isfinite(eps_planet) or not np.isfinite(d_eps_mono):
        return TIER_STAR

    # Negative ε_planet means the planet *took* energy away → star-dominated
    if eps_planet <= 0:
        return TIER_STAR

    abs_mono = abs(d_eps_mono)
    if abs_mono < 1e-30:
        # Δε_monopole ≈ 0 and ε_planet > 0 → planet-dominated
        return TIER_PLANET

    ratio = eps_planet / abs_mono

    if ratio > planet_threshold:
        return TIER_PLANET
    elif ratio >= hybrid_threshold:
        return TIER_HYBRID
    else:
        return TIER_STAR


def tier_ratio(analysis: Dict[str, Any]) -> Optional[float]:
    """Return the ε_planet / |Δε_monopole| ratio, or *None* if undefined."""
    eps_planet = analysis.get("energy_from_planet_orbit")
    d_eps_mono = analysis.get("delta_eps_monopole")
    if eps_planet is None or d_eps_mono is None:
        return None
    eps_planet = float(eps_planet)
    d_eps_mono = float(d_eps_mono)
    if not np.isfinite(eps_planet) or not np.isfinite(d_eps_mono):
        return None
    abs_mono = abs(d_eps_mono)
    if abs_mono < 1e-30:
        return float("inf") if eps_planet > 0 else 0.0
    return eps_planet / abs_mono


# ───────────────────────────────────────────────────────────────────
# Batch tiering
# ───────────────────────────────────────────────────────────────────

def tier_candidates(
    analyses: Sequence[Optional[Dict[str, Any]]],
    top_indices: Optional[Sequence[int]] = None,
    planet_threshold: float = 2.0,
    hybrid_threshold: float = 0.5,
) -> Dict[str, List[Tuple[int, int, Dict[str, Any]]]]:
    """Group valid candidates into tier buckets.

    Each bucket is a list of ``(rank, local_index, analysis)`` tuples sorted
    by scalar Δv (descending) within the tier.

    Parameters
    ----------
    analyses : sequence of dict or None
        ``analyses_best`` from the pipeline re-run phase.
    top_indices : sequence of int, optional
        MC indices corresponding to each analysis entry.
    planet_threshold, hybrid_threshold : float
        Tier boundary ratios.

    Returns
    -------
    dict
        ``{tier_label: [(rank, mc_idx, analysis), ...]}`` for each tier.
    """
    buckets: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {
        t: [] for t in TIER_ORDER
    }

    for local_i, ana in enumerate(analyses):
        if ana is None:
            continue
        mc_idx = int(top_indices[local_i]) if top_indices is not None else local_i
        tier = classify_candidate(ana, planet_threshold, hybrid_threshold)
        buckets[tier].append((0, mc_idx, ana))  # rank=0 placeholder

    # Sort each bucket by scalar Δv descending, then assign ranks
    for tier in TIER_ORDER:
        items = buckets[tier]
        items.sort(key=lambda x: float(x[2].get("delta_v", -np.inf)), reverse=True)
        buckets[tier] = [
            (rank, mc_idx, ana)
            for rank, (_, mc_idx, ana) in enumerate(items, start=1)
        ]

    return buckets


# ───────────────────────────────────────────────────────────────────
# Per-tier summary statistics
# ───────────────────────────────────────────────────────────────────

def tier_summary_stats(
    tiered: Dict[str, List[Tuple[int, int, Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-tier summary statistics.

    Returns
    -------
    dict
        ``{tier_label: {count, dv_min, dv_max, dv_median, ratio_min,
        ratio_max, ratio_median, top_mc_idx}}``
    """
    stats: Dict[str, Dict[str, Any]] = {}
    for tier in TIER_ORDER:
        items = tiered.get(tier, [])
        n = len(items)
        if n == 0:
            stats[tier] = {
                "count": 0,
                "dv_min": None, "dv_max": None, "dv_median": None,
                "ratio_min": None, "ratio_max": None, "ratio_median": None,
                "top_mc_idx": None,
            }
            continue

        dvs = np.array([float(a.get("delta_v", np.nan)) for _, _, a in items])
        dvs_fin = dvs[np.isfinite(dvs)]
        ratios = np.array([tier_ratio(a) or np.nan for _, _, a in items])
        ratios_fin = ratios[np.isfinite(ratios)]

        stats[tier] = {
            "count": n,
            "dv_min": float(np.min(dvs_fin)) if dvs_fin.size else None,
            "dv_max": float(np.max(dvs_fin)) if dvs_fin.size else None,
            "dv_median": float(np.median(dvs_fin)) if dvs_fin.size else None,
            "ratio_min": float(np.min(ratios_fin)) if ratios_fin.size else None,
            "ratio_max": float(np.max(ratios_fin)) if ratios_fin.size else None,
            "ratio_median": float(np.median(ratios_fin)) if ratios_fin.size else None,
            "top_mc_idx": items[0][1] if items else None,
        }

    return stats


# ───────────────────────────────────────────────────────────────────
# Convenience: annotate analyses in-place with tier label + ratio
# ───────────────────────────────────────────────────────────────────

def annotate_tiers(
    analyses: Sequence[Optional[Dict[str, Any]]],
    planet_threshold: float = 2.0,
    hybrid_threshold: float = 0.5,
) -> None:
    """Add ``'tier'`` and ``'tier_ratio'`` keys to each non-None analysis dict.

    Mutates the dicts in place so downstream consumers (report, CSV, notebook)
    can access tier info without re-computing.
    """
    for ana in analyses:
        if ana is None:
            continue
        ana["tier"] = classify_candidate(ana, planet_threshold, hybrid_threshold)
        ana["tier_ratio"] = tier_ratio(ana)
