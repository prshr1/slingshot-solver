"""
Comparison utilities for 2-body ↔ 3-body energy analysis.

All values in the canonical unit system: km-kg-s.
    Energy: km²/s² (≡ MJ/kg)
    ΔV    : km/s
"""

from typing import Dict, Any, Optional
import numpy as np
from .console import safe_print as print


def compare_2body_3body(
    energy_2body_star: Optional[float] = None,
    energy_2body_planet: Optional[float] = None,
    energy_3body: Optional[float] = None,
    dv_2body_star: Optional[float] = None,
    dv_2body_planet: Optional[float] = None,
    dv_3body: Optional[float] = None,
    dv_vec_2body_star: Optional[float] = None,
    dv_vec_2body_planet: Optional[float] = None,
    dv_vec_3body: Optional[float] = None,
    envelope_summary: Optional[str] = None,
    baseline_mode: str = "unknown",
) -> Dict[str, Any]:
    """Compare 2-body baselines with 3-body result.

    All energies in km²/s² (≡ MJ/kg), all ΔV in km/s.

    Parameters
    ----------
    energy_* : float, optional
        Specific orbital energy ε (max for 2-body, best for 3-body).
    dv_* : float, optional
        Scalar ΔV = |v_f| - |v_i|.
    dv_vec_* : float, optional
        Vector ΔV = |v_f - v_i| (consistent with TwoBodyScatter.deltaV_lab).
    envelope_summary : str, optional
        Human-readable envelope description (narrowed mode).
    baseline_mode : str
        "fixed" or "narrowed".

    Returns
    -------
    dict
        Comparison metrics with clearly labelled keys and units.
    """
    result: Dict[str, Any] = {
        "unit_energy": "km²/s² (≡ MJ/kg)",
        "unit_dv": "km/s",
        "baseline_mode": baseline_mode,
    }

    if energy_2body_star is not None:
        result["energy_2body_star"] = energy_2body_star
    if energy_2body_planet is not None:
        result["energy_2body_planet"] = energy_2body_planet
    if energy_3body is not None:
        result["energy_3body"] = energy_3body

    # Energy improvements
    if energy_3body is not None and energy_2body_star is not None:
        diff = energy_3body - energy_2body_star
        result["delta_E_vs_star"] = diff
        if energy_2body_star != 0:
            result["pct_vs_star"] = 100.0 * diff / abs(energy_2body_star)

    if energy_3body is not None and energy_2body_planet is not None:
        diff = energy_3body - energy_2body_planet
        result["delta_E_vs_planet"] = diff
        if energy_2body_planet != 0:
            result["pct_vs_planet"] = 100.0 * diff / abs(energy_2body_planet)

    # ΔV comparison
    if dv_2body_star is not None:
        result["dv_2body_star"] = dv_2body_star
    if dv_2body_planet is not None:
        result["dv_2body_planet"] = dv_2body_planet
    if dv_3body is not None:
        result["dv_3body"] = dv_3body
        if dv_2body_planet is not None:
            result["dv_improvement_vs_planet"] = dv_3body - dv_2body_planet
        if dv_2body_star is not None:
            result["dv_improvement_vs_star"] = dv_3body - dv_2body_star

    # Vector ΔV comparison
    if dv_vec_2body_star is not None:
        result["dv_vec_2body_star"] = dv_vec_2body_star
    if dv_vec_2body_planet is not None:
        result["dv_vec_2body_planet"] = dv_vec_2body_planet
    if dv_vec_3body is not None:
        result["dv_vec_3body"] = dv_vec_3body
        if dv_vec_2body_star is not None:
            diff = dv_vec_3body - dv_vec_2body_star
            result["dv_vec_improvement_vs_star"] = diff
            if dv_vec_2body_star != 0:
                result["dv_vec_pct_vs_star"] = 100.0 * diff / abs(dv_vec_2body_star)
        if dv_vec_2body_planet is not None:
            diff = dv_vec_3body - dv_vec_2body_planet
            result["dv_vec_improvement_vs_planet"] = diff
            if dv_vec_2body_planet != 0:
                result["dv_vec_pct_vs_planet"] = 100.0 * diff / abs(dv_vec_2body_planet)

    # Envelope metadata
    if envelope_summary is not None:
        result["envelope_summary"] = envelope_summary

    return result


def format_energy(value: float) -> str:
    """Format an energy value in km²/s² as a readable string.

    Since 1 km²/s² ≡ 1 MJ/kg, we display both.
    """
    if abs(value) >= 1e3:
        return f"{value:.2f} km²/s² ({value:.2f} MJ/kg)"
    else:
        return f"{value:.4f} km²/s² ({value:.4f} MJ/kg)"


def print_comparison(comp: Dict[str, Any]) -> None:
    """Pretty-print a comparison dict."""
    print("=" * 60)
    print("  2-BODY  vs  3-BODY  COMPARISON")
    print("  All energies in km²/s² ≡ MJ/kg")
    print("=" * 60)

    # --- Primary metric: ½|ΔV_vec|² (body-dependent) ---
    if "dv_vec_2body_star" in comp or "dv_vec_3body" in comp:
        print("\n--- ½|ΔV_vec|²  (scattering energy, body-dependent) ---")
    if "dv_vec_2body_star" in comp:
        e = 0.5 * comp['dv_vec_2body_star']**2
        print(f"  Star  2-body max ½|ΔV|²  : {format_energy(e)}")
    if "dv_vec_2body_planet" in comp:
        e = 0.5 * comp['dv_vec_2body_planet']**2
        print(f"  Planet 2-body max ½|ΔV|² : {format_energy(e)}")
    if "dv_vec_3body" in comp:
        e = 0.5 * comp['dv_vec_3body']**2
        print(f"  3-body best ½|ΔV|²       : {format_energy(e)}")

    # --- Legacy ε = ½v∞² (body-independent info) ---
    if "energy_2body_star" in comp or "energy_2body_planet" in comp:
        print("\n--- ε = ½v∞²  (approach KE — same for any body at same speed) ---")
    if "energy_2body_star" in comp:
        print(f"  Star  baseline ε  : {format_energy(comp['energy_2body_star'])}")
    if "energy_2body_planet" in comp:
        print(f"  Planet baseline ε : {format_energy(comp['energy_2body_planet'])}")
    if "energy_3body" in comp:
        print(f"  3-body best ε     : {format_energy(comp['energy_3body'])}")

    print("-" * 60)

    if "dv_3body" in comp:
        print(f"\n  ΔV scalar (3-body best) : {comp['dv_3body']:.2f} km/s")
    if "dv_improvement_vs_planet" in comp:
        sign = "+" if comp["dv_improvement_vs_planet"] >= 0 else ""
        print(f"  ΔV improvement (planet)  : {sign}{comp['dv_improvement_vs_planet']:.2f} km/s")
    if "dv_improvement_vs_star" in comp:
        sign = "+" if comp["dv_improvement_vs_star"] >= 0 else ""
        print(f"  ΔV improvement (star)    : {sign}{comp['dv_improvement_vs_star']:.2f} km/s")

    # Scalar ΔV comparison (primary metric for interstellar slingshot research)
    if "dv_2body_star" in comp or "dv_2body_planet" in comp:
        print("\n--- Scalar ΔV (|v_f| - |v_i|) ---")
    if "dv_2body_star" in comp:
        print(f"  Star  2-body max scalar ΔV    : {comp['dv_2body_star']:.2f} km/s")
    if "dv_2body_planet" in comp:
        print(f"  Planet 2-body max scalar ΔV   : {comp['dv_2body_planet']:.2f} km/s")
    if "dv_3body" in comp:
        print(f"  3-body best scalar ΔV         : {comp['dv_3body']:.2f} km/s")

    # Vector ΔV section
    if "dv_vec_2body_star" in comp or "dv_vec_3body" in comp:
        print("\n--- Vector ΔV (|v_f - v_i|) ---")
    if "dv_vec_2body_star" in comp:
        print(f"  Star  2-body max |ΔV_vec|  : {comp['dv_vec_2body_star']:.2f} km/s")
    if "dv_vec_2body_planet" in comp:
        print(f"  Planet 2-body max |ΔV_vec| : {comp['dv_vec_2body_planet']:.2f} km/s")
    if "dv_vec_3body" in comp:
        print(f"  3-body best |ΔV_vec|       : {comp['dv_vec_3body']:.2f} km/s")
    if "dv_vec_pct_vs_star" in comp:
        sign = "+" if comp["dv_vec_pct_vs_star"] >= 0 else ""
        print(f"  vs star baseline           : {sign}{comp['dv_vec_pct_vs_star']:.1f}%")
    if "dv_vec_pct_vs_planet" in comp:
        sign = "+" if comp["dv_vec_pct_vs_planet"] >= 0 else ""
        print(f"  vs planet baseline         : {sign}{comp['dv_vec_pct_vs_planet']:.1f}%")

    # Envelope info
    if "envelope_summary" in comp:
        print(f"\n{comp['envelope_summary']}")
    if "baseline_mode" in comp:
        print(f"  [baseline mode: {comp['baseline_mode']}]")

    # Planet-as-primary benchmark (for slingshot research)
    if "dv_vec_3body" in comp and "dv_vec_2body_planet" in comp:
        e_3b = 0.5 * comp['dv_vec_3body']**2
        e_pl = 0.5 * comp['dv_vec_2body_planet']**2
        ratio = e_3b / e_pl if e_pl > 0 else float('inf')
        print(f"\n  ★ Planet slingshot amplification: {ratio:.1f}× planet-only ceiling")
        print(f"    (3-body ½|ΔV|² = {e_3b:.1f} vs planet-only = {e_pl:.1f} km²/s²)")

    print("=" * 60)
