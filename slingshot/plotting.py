"""
Visualization and plotting utilities.

Includes:
  - 3-body trajectory / MC summary plots (original)
  - 5 new diagnostic plots: star proximity, planet-frame diagnostics,
    multi-candidate overlay, rejection breakdown, parameter correlations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Dict, Any, List
from pathlib import Path

from .constants import G_KM, M_SUN, R_SUN, M_JUP, R_JUP


def plot_best_candidate_with_bodies(
    sol: object,
    analysis: Optional[Dict[str, Any]] = None,
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    R_star: Optional[float] = None,
    R_p: Optional[float] = None,
    figsize: tuple = (10, 10),
    title: str = "Slingshot Trajectory",
) -> plt.Figure:
    """
    Plot trajectory with scaled star and planet circles.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    analysis : dict, optional
        Analysis results to display
    m_star, m_p : float, optional
        Masses for radius scaling
    R_star, R_p : float, optional
        Radii (km)
    figsize : tuple
        Figure size
    title : str
        Plot title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    if R_p is None:
        R_p = R_JUP
    
    # Estimate star radius if not provided
    if R_star is None:
        R_star = R_SUN * np.sqrt(m_star / M_SUN)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if sol is None:
        ax.text(0.5, 0.5, "No valid solution", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return fig
    
    y = sol.y
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    x3 = y[8]
    y3 = y[9]
    
    # Plot trajectories
    ax.plot(xs, ys, label="Star", lw=1.5, alpha=0.7, color='gold')
    ax.plot(xp, yp, label="Planet", lw=1.5, alpha=0.7, color='darkred')
    ax.plot(x3, y3, label="Satellite", lw=2.0, alpha=0.8, color='blue')
    
    # Initial position circles
    star_circle = plt.Circle((xs[0], ys[0]), R_star, fill=True, facecolor='gold',
                             alpha=0.6, edgecolor='orange', linewidth=2)
    planet_circle = plt.Circle((xp[0], yp[0]), R_p, fill=True, facecolor='darkred',
                               alpha=0.6, edgecolor='red', linewidth=2)
    ax.add_artist(star_circle)
    ax.add_artist(planet_circle)
    
    # Final positions
    ax.plot(xs[-1], ys[-1], 'o', color='gold', markersize=8)
    ax.plot(xp[-1], yp[-1], 'o', color='darkred', markersize=8)
    ax.plot(x3[-1], y3[-1], 'o', color='blue', markersize=8)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (km)', fontsize=11)
    ax.set_ylabel('y (km)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Analysis text box
    if analysis:
        textlines = []
        if 'delta_v' in analysis:
            textlines.append(f"Δv = {analysis['delta_v']:.2f} km/s")
        if 'deflection' in analysis:
            textlines.append(f"Deflection = {analysis['deflection']:.1f}°")
        if 'r_min' in analysis:
            textlines.append(f"r_min = {analysis['r_min']:.0f} km")
        if 'unbound_f' in analysis:
            textlines.append(f"Unbound: {analysis['unbound_f']}")
        
        textstr = '\n'.join(textlines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    return fig


def plot_mc_summary(
    mc: Dict[str, Any],
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot Monte Carlo summary: scatter of deflection vs Δv, histogram of deflection/180°.
    
    Parameters
    ----------
    mc : dict
        Monte Carlo results
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ok = mc["ok"]
    dv = mc["delta_v"][ok]
    df = mc["deflection"][ok]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter: deflection vs Δv
    ax = axes[0]
    ax.scatter(df, dv, s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel("Deflection angle (deg)", fontsize=11)
    ax.set_ylabel("Δv (km/s)", fontsize=11)
    ax.set_title("Slingshot outcomes (successful)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Histogram: deflection fraction
    ax = axes[1]
    deflection_frac = df / 180.0
    ax.hist(deflection_frac, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Deflection / 180°", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Deflection distribution", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_velocity_phase_space(
    sol: object,
    title_prefix: str = "",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot velocity phase space: (vx, vy) and (v_radial, v_normal).
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    title_prefix : str
        Prefix for plot titles
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sol is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No solution available", ha='center', va='center',
                transform=ax.transAxes)
        return fig
    
    y = sol.y
    
    # Barycentric satellite velocity
    vx_sat = y[10]
    vy_sat = y[11]
    
    # Planet-frame velocity
    vxp = y[6]
    vyp = y[7]
    dvx = vx_sat - vxp
    dvy = vy_sat - vyp
    
    # Radial and normal components (planet frame)
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    dx = xs - xp
    dy = ys - yp
    r = np.hypot(dx, dy)
    
    # Unit radial (planet -> satellite)
    erx = dx / (r + 1e-10)
    ery = dy / (r + 1e-10)
    
    # Unit normal (tangential)
    enx = -ery
    eny = erx
    
    v_rad = dvx * erx + dvy * ery
    v_norm = dvx * enx + dvy * eny
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # (vx, vy) phase space
    ax = axes[0]
    ax.plot(vx_sat, vy_sat, lw=1, alpha=0.8)
    ax.set_xlabel('vx (km/s)', fontsize=11)
    ax.set_ylabel('vy (km/s)', fontsize=11)
    ax.set_title(f'{title_prefix} Satellite (vx, vy) phase space', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # (v_radial, v_normal) phase space
    ax = axes[1]
    ax.plot(v_rad, v_norm, lw=1, alpha=0.8)
    ax.set_xlabel('v_radial (km/s)', fontsize=11)
    ax.set_ylabel('v_normal (km/s)', fontsize=11)
    ax.set_title(f'{title_prefix} Radial vs Normal velocity', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ===================================================================
# New diagnostic plots — 3-body Monte Carlo analysis
# ===================================================================

def plot_star_proximity_distribution(
    mc: Dict[str, Any],
    R_star_km: float,
    clearance_Rstar: Optional[float] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot histogram of closest-approach distance to the star.

    Helps diagnose whether top candidates are planet-dominated
    (slingshot) or star-dominated (stellar flyby / grazing orbit).

    Parameters
    ----------
    mc : dict
        Monte Carlo results (must contain ``r_star_min``).
    R_star_km : float
        Stellar radius in km.
    clearance_Rstar : float, optional
        If set, draw a vertical line at clearance × R_star.
    """
    r_star_all = mc.get("r_star_min", None)
    if r_star_all is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "r_star_min not recorded", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    ok = mc["ok"]
    r_ok = r_star_all[ok]
    r_ok = r_ok[np.isfinite(r_ok)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # — All particles (that have a valid analysis) —
    r_all_valid = r_star_all[np.isfinite(r_star_all)]
    ax = axes[0]
    ax.hist(r_all_valid / R_star_km, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("r_star_min / R★", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("All particles — star closest approach", fontsize=11, fontweight="bold")
    if clearance_Rstar is not None:
        ax.axvline(clearance_Rstar, color="red", ls="--", lw=2,
                   label=f"Filter: {clearance_Rstar:.1f} R★")
        ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # — Successful particles only —
    ax = axes[1]
    if r_ok.size > 0:
        ax.hist(r_ok / R_star_km, bins=40, alpha=0.7, color="darkorange", edgecolor="black")
    ax.set_xlabel("r_star_min / R★", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Successful particles — star closest approach", fontsize=11, fontweight="bold")
    if clearance_Rstar is not None:
        ax.axvline(clearance_Rstar, color="red", ls="--", lw=2,
                   label=f"Filter: {clearance_Rstar:.1f} R★")
        ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "star_proximity_distribution.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_planet_frame_diagnostics(
    analyses: List[Dict[str, Any]],
    R_p_km: float,
    R_star_km: float,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot planet-frame diagnostic quantities for re-run candidates.

    Shows whether ΔV comes from the planet's orbital KE (slingshot)
    vs deep stellar potential (star-dominated).

    Parameters
    ----------
    analyses : list of dict
        Each dict from ``analyze_trajectory(frame='barycentric')``.
    """
    dv_pf = []  # planet-frame |Δv|
    defl_pf = []  # planet-frame deflection
    e_planet = []  # energy from planet orbit
    r_min_star = []  # star closest approach

    for ana in analyses:
        if ana is None:
            continue
        dv_pf.append(ana.get("delta_v_planet_frame", np.nan))
        defl_pf.append(ana.get("planet_deflection_deg", np.nan))
        e_planet.append(ana.get("energy_from_planet_orbit", np.nan))
        enc = ana.get("encounter")
        r_min_star.append(enc.r_star_min if enc and enc.r_star_min else np.nan)

    dv_pf = np.array(dv_pf)
    defl_pf = np.array(defl_pf)
    e_planet = np.array(e_planet)
    r_min_star = np.array(r_min_star)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # planet-frame Δv (should be ~0 for pure 2-body flyby)
    ax = axes[0, 0]
    ax.bar(range(len(dv_pf)), dv_pf, color="teal", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Δv (planet frame) [km/s]")
    ax.set_title("Planet-frame speed change (should ≈ 0 for pure flyby)")

    # planet-frame deflection
    ax = axes[0, 1]
    ax.bar(range(len(defl_pf)), defl_pf, color="darkorange", alpha=0.7)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Deflection (°)")
    ax.set_title("Planet-frame deflection")

    # energy from planet orbit
    ax = axes[1, 0]
    ax.bar(range(len(e_planet)), e_planet, color="royalblue", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("ΔE from planet orbit [km²/s²]")
    ax.set_title("Energy extracted from planet orbital KE")

    # star closest approach
    ax = axes[1, 1]
    vals = r_min_star / R_star_km
    ax.bar(range(len(vals)), vals, color="crimson", alpha=0.7)
    ax.axhline(1.0, color="gold", ls="--", lw=1.5, label="R★ surface")
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("r_star_min / R★")
    ax.set_title("Star closest approach per candidate")
    ax.legend()

    fig.suptitle("Planet-frame diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "planet_frame_diagnostics.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_multi_candidate_overlay(
    sols: list,
    analyses: List[Dict[str, Any]],
    m_star: float,
    m_p: float,
    R_star: float,
    R_p: float,
    top_n: int = 5,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Overlay the top-N best candidate trajectories on a single figure.

    Parameters
    ----------
    sols : list
        ODE solutions for best candidates.
    analyses : list of dict
        Corresponding analysis dicts.
    top_n : int
        Number to overlay (uses first ``top_n`` from lists).
    """
    n = min(top_n, len(sols))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    fig, ax = plt.subplots(figsize=(11, 11))

    for k in range(n):
        sol = sols[k]
        if sol is None:
            continue
        y = sol.y
        color = cmap[k]
        label = f"#{k+1}"
        if k < len(analyses) and analyses[k]:
            dv = analyses[k].get("delta_v", np.nan)
            label += f" Δv={dv:+.1f}"
        ax.plot(y[8], y[9], lw=1.5, alpha=0.8, color=color, label=label)

    # Draw star and planet at initial positions (from first valid sol)
    for sol in sols:
        if sol is not None:
            y0 = sol.y
            star_c = Circle((y0[0, 0], y0[1, 0]), R_star, fc="gold", ec="orange",
                            alpha=0.5, lw=1.5)
            planet_c = Circle((y0[4, 0], y0[5, 0]), R_p, fc="darkred", ec="red",
                              alpha=0.5, lw=1.5)
            ax.add_artist(star_c)
            ax.add_artist(planet_c)
            break

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (km)", fontsize=11)
    ax.set_ylabel("y (km)", fontsize=11)
    ax.set_title(f"Top-{n} candidate trajectories", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "multi_candidate_overlay.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_rejection_breakdown(
    mc: Dict[str, Any],
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Bar chart of rejection/failure reasons from Monte Carlo results.

    Parameters
    ----------
    mc : dict
        Monte Carlo results (must contain ``results`` list).
    """
    results_list = mc.get("results", [])
    reason_counts: Dict[str, int] = {}
    for r in results_list:
        reason = r.get("reason", "")
        if r["ok"]:
            reason = "success"
        elif not reason:
            reason = "unknown_fail"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    labels = sorted(reason_counts, key=reason_counts.get, reverse=True)
    counts = [reason_counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if l == "success" else "#d62728" if "collision" in l
              else "#ff7f0e" for l in labels]
    ax.barh(labels, counts, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Monte Carlo rejection breakdown", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Percentage annotations
    total = sum(counts)
    for i, (c, lbl) in enumerate(zip(counts, labels)):
        ax.text(c + total * 0.005, i, f"{100*c/total:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "rejection_breakdown.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_parameter_correlations(
    mc: Dict[str, Any],
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Scatter matrix of key MC outcome metrics for successful particles.

    Plots: Δv vs deflection, r_min vs Δv, r_star_min vs deflection,
    delta_v_vec vs delta_v.

    Parameters
    ----------
    mc : dict
        Monte Carlo results.
    """
    ok = mc["ok"]
    dv = mc["delta_v"][ok]
    defl = mc["deflection"][ok]
    rmin = mc["r_min"][ok]
    dv_vec = mc.get("delta_v_vec", np.full_like(mc["delta_v"], np.nan))[ok]
    r_star = mc.get("r_star_min", np.full_like(mc["delta_v"], np.nan))[ok]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Δv vs deflection
    ax = axes[0, 0]
    ax.scatter(defl, dv, s=12, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Deflection (°)")
    ax.set_ylabel("Δv (km/s)")
    ax.set_title("Δv vs Deflection")
    ax.grid(True, alpha=0.3)

    # r_min vs Δv
    ax = axes[0, 1]
    ax.scatter(rmin, dv, s=12, alpha=0.5, c="darkorange", edgecolors="none")
    ax.set_xlabel("r_min (km)")
    ax.set_ylabel("Δv (km/s)")
    ax.set_title("Δv vs Periapsis")
    ax.grid(True, alpha=0.3)

    # r_star_min vs deflection
    ax = axes[1, 0]
    mask = np.isfinite(r_star)
    if mask.any():
        ax.scatter(r_star[mask], defl[mask], s=12, alpha=0.5, c="crimson", edgecolors="none")
    ax.set_xlabel("r_star_min (km)")
    ax.set_ylabel("Deflection (°)")
    ax.set_title("Deflection vs Star proximity")
    ax.grid(True, alpha=0.3)

    # |ΔV_vec| vs scalar Δv
    ax = axes[1, 1]
    mask = np.isfinite(dv_vec)
    if mask.any():
        ax.scatter(dv[mask], dv_vec[mask], s=12, alpha=0.5, c="teal", edgecolors="none")
        # Reference line y = x
        lim = max(abs(dv[mask]).max(), abs(dv_vec[mask]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y = x")
        ax.legend(fontsize=9)
    ax.set_xlabel("Scalar Δv = |v_f| - |v_i| (km/s)")
    ax.set_ylabel("|ΔV_vec| = |v_f - v_i| (km/s)")
    ax.set_title("Vector vs Scalar ΔV")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Parameter correlations — successful particles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "parameter_correlations.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_energy_cdf(
    mc: Dict[str, Any],
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """CDF of ½|ΔV_vec|² for successful particles.

    Parameters
    ----------
    mc : dict
        Monte Carlo results.
    """
    ok = mc["ok"]
    dv_vec = mc.get("delta_v_vec", np.full_like(mc["delta_v"], np.nan))[ok]
    dv_vec = dv_vec[np.isfinite(dv_vec)]
    energy = 0.5 * dv_vec ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    if energy.size > 0:
        sorted_e = np.sort(energy)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf, lw=2, color="royalblue")
        ax.fill_between(sorted_e, 0, cdf, alpha=0.15, color="royalblue")

        # Percentile markers
        for pct in [50, 90, 95, 99]:
            val = np.percentile(sorted_e, pct)
            ax.axhline(pct / 100, color="gray", ls=":", alpha=0.5)
            ax.axvline(val, color="gray", ls=":", alpha=0.5)
            ax.text(val, pct / 100 + 0.02, f"P{pct}: {val:.1f}", fontsize=8)

    ax.set_xlabel("½|ΔV_vec|² (km²/s²)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("Energy CDF — ½|ΔV_vec|² distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "energy_cdf.png", dpi=dpi, bbox_inches="tight")
    return fig
