"""
Auto-generated markdown run report.

Produces a ``REPORT.md`` summarising a pipeline run:
system parameters, MC statistics, best-candidate metrics,
2-body vs 3-body comparison, and a listing of saved plots.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

from .config import FullConfig
from .constants import R_SUN, R_JUP


def generate_run_report(
    output_dir: Path,
    cfg: FullConfig,
    mc: Dict[str, Any],
    analyses_best: List[Optional[Dict]],
    best: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
    narrowed: Optional[Dict[str, Any]] = None,
    saved_plots: Optional[List[str]] = None,
) -> str:
    """Generate a markdown report and write it to ``output_dir/REPORT.md``.

    Returns the markdown string.
    """
    lines: List[str] = []

    def h1(t):
        lines.append(f"# {t}\n")

    def h2(t):
        lines.append(f"## {t}\n")

    def h3(t):
        lines.append(f"### {t}\n")

    def p(t):
        lines.append(f"{t}\n")

    def table_row(*cols):
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    # --- Header ---
    h1(f"Slingshot Pipeline Report — {cfg.system.name}")
    p(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"**Output:** `{output_dir}`\n")

    # --- System ---
    h2("System Parameters")
    R_star = cfg.system.R_star_Rsun * R_SUN
    R_p = cfg.system.R_planet_Rjup * R_JUP
    p(f"| Parameter | Value |")
    p(f"|-----------|-------|")
    p(f"| Star mass | {cfg.system.M_star_Msun:.2f} M☉ |")
    p(f"| Star radius | {cfg.system.R_star_Rsun:.2f} R☉ ({R_star:.0f} km) |")
    p(f"| Planet mass | {cfg.system.M_planet_Mjup:.1f} M♃ |")
    p(f"| Planet radius | {cfg.system.R_planet_Rjup:.3f} R♃ ({R_p:.0f} km) |")
    p(f"| Orbit | {cfg.system.a_planet_AU:.4f} AU |")
    p("")

    # --- Numerical ---
    h2("Numerical Configuration")
    p(f"| Parameter | Value |")
    p(f"|-----------|-------|")
    p(f"| ODE method | {cfg.numerical.ode_method} |")
    p(f"| rtol / atol | {cfg.numerical.rtol} / {cfg.numerical.atol} |")
    p(f"| Softening | {cfg.numerical.softening_km:.1f} km |")
    p(f"| Star filter | {cfg.numerical.star_min_clearance_Rstar} R★ |")
    p(f"| r_far_factor | {cfg.numerical.r_far_factor} |")
    p(f"| Escape factor | {cfg.numerical.escape_radius_factor} |")
    p("")

    # --- MC summary ---
    h2("Monte Carlo Sweep")
    ok = mc["ok"]
    ok_count = int(ok.sum())
    N = len(ok)
    dv_ok = mc["delta_v"][ok]
    defl_ok = mc["deflection"][ok]

    p(f"- **Particles:** {N:,}")
    p(f"- **Successful:** {ok_count:,} ({100*ok_count/N:.1f}%)")
    if dv_ok.size > 0:
        p(f"- **Δv range:** [{dv_ok.min():.2f}, {dv_ok.max():.2f}] km/s")
        p(f"- **Δv mean ± σ:** {dv_ok.mean():.2f} ± {dv_ok.std():.2f} km/s")
        p(f"- **Deflection range:** [{defl_ok.min():.1f}°, {defl_ok.max():.1f}°]")
    p("")

    # Rejection breakdown
    h3("Rejection Breakdown")
    reasons: Dict[str, int] = {}
    for r in mc.get("results", []):
        if r["ok"]:
            reasons["success"] = reasons.get("success", 0) + 1
        else:
            reason = r.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
    p("| Reason | Count | % |")
    p("|--------|-------|---|")
    for reason in sorted(reasons, key=reasons.get, reverse=True):
        cnt = reasons[reason]
        p(f"| {reason} | {cnt} | {100*cnt/N:.1f}% |")
    p("")

    # --- Best candidates ---
    h2("Best Candidates")
    ba = best.get("best_ana")
    if ba:
        h3("Best by scalar Δv")
        p(f"- MC index: **{best['best_idx']}**")
        p(f"- Δv: **{ba['delta_v']:.4f}** km/s ({ba['delta_v_pct']:+.1f}%)")
        p(f"- |ΔV_vec|: {ba.get('delta_v_vec', 0):.4f} km/s")
        p(f"- ½|ΔV_vec|²: {ba.get('energy_half_dv_vec_sq', 0):.4f} km²/s²")
        p(f"- Deflection: {ba['deflection']:.1f}°")
        p(f"- r_min (planet): {ba['r_min']:.0f} km ({ba['r_min']/R_p:.1f} R_p)")
        enc = ba.get("encounter")
        if enc and enc.r_star_min:
            p(f"- r_min (star): {enc.r_star_min:.0f} km ({enc.r_star_min/R_star:.1f} R★)")
        dvpf = ba.get("delta_v_planet_frame")
        if dvpf is not None:
            p(f"- Δv planet frame: {dvpf:.4f} km/s")
            p(f"- Energy from planet orbit: {ba.get('energy_from_planet_orbit', 0):.4e} km²/s²")
        p("")

    bva = best.get("best_vec_ana")
    if bva and best.get("best_vec_idx") != best.get("best_idx"):
        h3("Best by ½|ΔV_vec|²")
        p(f"- MC index: **{best['best_vec_idx']}**")
        p(f"- |ΔV_vec|: **{bva.get('delta_v_vec', 0):.4f}** km/s")
        p(f"- ½|ΔV_vec|²: **{bva.get('energy_half_dv_vec_sq', 0):.4f}** km²/s²")
        p(f"- Deflection: {bva['deflection']:.1f}°")
        p("")

    # --- Candidates table ---
    valid_analyses = [(i, a) for i, a in enumerate(analyses_best) if a is not None]
    if valid_analyses:
        h3("All Re-run Candidates")
        p("| Rank | Δv (km/s) | |ΔV_vec| | Deflection (°) | r_min (km) | r_star (R★) |")
        p("|------|-----------|---------|----------------|------------|-------------|")
        for i, a in valid_analyses:
            enc = a.get("encounter")
            r_star_val = enc.r_star_min / R_star if enc and enc.r_star_min else float("nan")
            p(f"| {i+1} | {a['delta_v']:.2f} | {a.get('delta_v_vec',0):.2f} "
              f"| {a['deflection']:.1f} | {a['r_min']:.0f} | {r_star_val:.1f} |")
        p("")

    # --- Comparison ---
    if comparison:
        h2("2-Body vs 3-Body Comparison")
        p(f"- Baseline mode: **{comparison.get('baseline_mode', '?')}**")
        if "dv_vec_3body" in comparison:
            p(f"- 3-body best |ΔV_vec|: {comparison['dv_vec_3body']:.2f} km/s")
        if "dv_vec_2body_star" in comparison:
            p(f"- Star 2-body max |ΔV_vec|: {comparison['dv_vec_2body_star']:.2f} km/s")
        if "dv_vec_2body_planet" in comparison:
            p(f"- Planet 2-body max |ΔV_vec|: {comparison['dv_vec_2body_planet']:.2f} km/s")
        if "dv_vec_pct_vs_star" in comparison:
            p(f"- vs Star: {comparison['dv_vec_pct_vs_star']:+.1f}%")
        if "dv_vec_pct_vs_planet" in comparison:
            p(f"- vs Planet: {comparison['dv_vec_pct_vs_planet']:+.1f}%")
        # Amplification
        if "dv_vec_3body" in comparison and "dv_vec_2body_planet" in comparison:
            e3 = 0.5 * comparison["dv_vec_3body"] ** 2
            ep = 0.5 * comparison["dv_vec_2body_planet"] ** 2
            if ep > 0:
                p(f"- **Planet slingshot amplification:** {e3/ep:.1f}×")
        p("")

    # --- Saved plots ---
    if saved_plots:
        h2("Saved Plots")
        for sp in saved_plots:
            name = Path(sp).name
            p(f"- [{name}]({name})")
        p("")

    # --- Footer ---
    p("---")
    from . import __version__
    p(f"*slingshot-solver v{__version__}*")

    report_text = "\n".join(lines)

    # Write to file
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    return report_text
