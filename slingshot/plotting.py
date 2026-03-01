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
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from .constants import G_KM, M_SUN, R_SUN, M_JUP, R_JUP, AU_KM


def _enc_field(enc: Any, key: str, default: Any = np.nan) -> Any:
    """Access encounter field from dataclass-like object or dict."""
    if enc is None:
        return default
    if isinstance(enc, dict):
        return enc.get(key, default)
    return getattr(enc, key, default)


def save_subplot_panels(
    fig: plt.Figure,
    save_dir: Path,
    base_name: str,
    dpi: int = 150,
) -> List[str]:
    """Export each subplot axis of a figure as an individual PNG panel.

    Colorbar-only axes are skipped. Returns saved file paths.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = [
        ax for ax in fig.axes
        if ax.get_visible() and ax.get_label() != "<colorbar>"
    ]
    if len(axes) <= 1:
        return []

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    saved: List[str] = []
    for i, ax in enumerate(axes, start=1):
        title = ax.get_title() or f"panel_{i:02d}"
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
        filename = f"{base_name}_panel_{i:02d}_{slug}.png" if slug else f"{base_name}_panel_{i:02d}.png"
        out_path = out_dir / filename

        bbox = ax.get_tightbbox(renderer).expanded(1.05, 1.12)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
        saved.append(str(out_path))

    return saved


def plot_sampling_parameter_distributions(
    mc: Dict[str, Any],
    cfg: Optional[Any] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> Dict[str, plt.Figure]:
    """Plot proposal-parameter distributions with config cutoffs overlaid.

    Returns a mapping: output filename -> figure.
    """
    params = mc.get("sampling_params", None)
    ok = np.asarray(mc.get("ok", []), dtype=bool)
    mode = str(mc.get("sampling_mode", "")).lower()

    # Backward-compatible fallback for older MC artifacts that do not
    # include explicit sampling_params metadata.
    if not isinstance(params, dict) or not params:
        sat = np.asarray(mc.get("sat_states", np.empty((0, 4))), dtype=float)
        if sat.ndim != 2 or sat.shape[1] < 4 or sat.shape[0] == 0:
            return {}

        if mode == "barycentric":
            x = sat[:, 0]
            y = sat[:, 1]
            vx = sat[:, 2]
            vy = sat[:, 3]
            vmag = np.hypot(vx, vy)
            with np.errstate(invalid="ignore", divide="ignore"):
                b_km = np.abs(x * vy - y * vx) / vmag
            r = np.hypot(x, y)
            with np.errstate(invalid="ignore", divide="ignore"):
                rx = x / r
                ry = y / r
                tx = -ry
                ty = rx
                inward = -(vx * rx + vy * ry)
                tangent = vx * tx + vy * ty
            params = {
                "v_mag_kms": vmag,
                "impact_param_AU": b_km / AU_KM,
                "angle_in_deg": np.degrees(np.arctan2(tangent, inward)),
                "azimuth_deg": np.degrees(np.arctan2(y, x)),
                "r_init_AU": r / AU_KM,
            }
        elif mode == "planet":
            Y_sp0 = np.asarray(mc.get("Y_sp0", np.empty(8)), dtype=float).ravel()
            if Y_sp0.size < 8:
                return {}
            xp, yp, vxp, vyp = float(Y_sp0[4]), float(Y_sp0[5]), float(Y_sp0[6]), float(Y_sp0[7])
            dx = sat[:, 0] - xp
            dy = sat[:, 1] - yp
            dvx = sat[:, 2] - vxp
            dvy = sat[:, 3] - vyp
            r = np.hypot(dx, dy)
            vrel = np.hypot(dvx, dvy)
            R_p = float(mc.get("R_p", np.nan))
            with np.errstate(invalid="ignore", divide="ignore"):
                rx = dx / r
                ry = dy / r
                tx = -ry
                ty = rx
                inward = -(dvx * rx + dvy * ry)
                tangent = dvx * tx + dvy * ty
                r0_Rp = r / R_p if np.isfinite(R_p) and R_p > 0 else np.full_like(r, np.nan)
            params = {
                "r0_Rp": r0_Rp,
                "v_rel_kms": vrel,
                "alpha_deg": np.degrees(np.arctan2(tangent, inward)),
                "theta_deg": np.degrees(np.arctan2(dy, dx)),
            }
        else:
            return {}

    sampling_cfg = getattr(cfg, "sampling", None) if cfg is not None else None

    if mode == "barycentric":
        specs = [
            ("v_mag_kms", "Initial speed v_mag", "km/s",
             getattr(sampling_cfg, "v_mag_min_kms", None),
             getattr(sampling_cfg, "v_mag_max_kms", None)),
            ("impact_param_AU", "Impact parameter b", "AU",
             getattr(sampling_cfg, "impact_param_min_AU", None),
             getattr(sampling_cfg, "impact_param_max_AU", None)),
            ("angle_in_deg", "Incoming angle", "deg",
             getattr(sampling_cfg, "angle_in_min_deg", None),
             getattr(sampling_cfg, "angle_in_max_deg", None)),
            ("r_init_AU", "Initial radius r_init", "AU",
             getattr(sampling_cfg, "r_init_AU", None),
             getattr(sampling_cfg, "r_init_AU", None)),
        ]
    elif mode == "planet":
        specs = [
            ("r0_Rp", "Initial radius r0 / R_p", "R_p",
             getattr(sampling_cfg, "r_min_factor", None),
             getattr(sampling_cfg, "r_max_factor", None)),
            ("v_rel_kms", "Relative speed to planet", "km/s",
             getattr(sampling_cfg, "v_rel_min_kms", None),
             getattr(sampling_cfg, "v_rel_max_kms", None)),
        ]
    else:
        specs = [(k, k, "", None, None) for k in params.keys()]

    figs: Dict[str, plt.Figure] = {}
    for key, title, unit, lower, upper in specs:
        if key not in params:
            continue

        arr_all = np.asarray(params[key], dtype=float)
        finite = np.isfinite(arr_all)
        if not finite.any():
            continue

        arr = arr_all[finite]
        if ok.size == arr_all.size:
            ok_finite = ok[finite]
            arr_ok = arr[ok_finite]
        else:
            arr_ok = np.array([], dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(arr, bins=40, alpha=0.55, edgecolor="black", label=f"All (n={arr.size})")
        if arr_ok.size > 0:
            ax.hist(arr_ok, bins=30, alpha=0.55, edgecolor="black", label=f"Successful (n={arr_ok.size})")

        if lower is not None and np.isfinite(lower):
            ax.axvline(float(lower), color="tab:red", ls="--", lw=1.5, label="Lower cutoff")
        if upper is not None and np.isfinite(upper):
            ax.axvline(float(upper), color="tab:purple", ls="--", lw=1.5, label="Upper cutoff")

        xlabel = f"{title} [{unit}]" if unit else title
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Sampling distribution: {title}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        fname = f"sampling_distribution_{key}.png"
        figs[fname] = fig
        if save_dir is not None:
            fig.savefig(Path(save_dir) / fname, dpi=dpi, bbox_inches="tight")

    return figs


def _pareto_front_mask_max(values: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for max-oriented objective matrix."""
    n = values.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


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


def plot_mc_summary_individual(
    mc: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return MC summary as standalone single-panel figures."""
    ok = mc["ok"]
    dv = mc["delta_v"][ok]
    df = mc["deflection"][ok]
    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df, dv, s=20, alpha=0.6, edgecolors="none")
    ax.set_xlabel("Deflection angle (deg)", fontsize=11)
    ax.set_ylabel("Delta-v (km/s)", fontsize=11)
    ax.set_title("Slingshot outcomes (successful)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["mc_summary_slingshot_outcomes.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    deflection_frac = df / 180.0
    ax.hist(deflection_frac, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Deflection / 180 deg", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Deflection distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["mc_summary_deflection_distribution.png"] = fig

    return figs


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


def plot_velocity_phase_space_individual(
    sol: object,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return velocity phase-space diagnostics as standalone figures."""
    if sol is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No solution available", ha="center", va="center",
                transform=ax.transAxes)
        return {"velocity_phase_space_unavailable.png": fig}

    y = sol.y
    vx_sat = y[10]
    vy_sat = y[11]
    vxp = y[6]
    vyp = y[7]
    dvx = vx_sat - vxp
    dvy = vy_sat - vyp
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    dx = xs - xp
    dy = ys - yp
    r = np.hypot(dx, dy)
    erx = dx / (r + 1e-10)
    ery = dy / (r + 1e-10)
    enx = -ery
    eny = erx
    v_rad = dvx * erx + dvy * ery
    v_norm = dvx * enx + dvy * eny

    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(vx_sat, vy_sat, lw=1, alpha=0.8)
    ax.set_xlabel("vx (km/s)", fontsize=11)
    ax.set_ylabel("vy (km/s)", fontsize=11)
    ax.set_title(f"{title_prefix} Satellite (vx, vy) phase space".strip(), fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    figs["velocity_phase_space_vx_vy.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(v_rad, v_norm, lw=1, alpha=0.8)
    ax.set_xlabel("v_radial (km/s)", fontsize=11)
    ax.set_ylabel("v_normal (km/s)", fontsize=11)
    ax.set_title(f"{title_prefix} Radial vs Normal velocity".strip(), fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["velocity_phase_space_radial_normal.png"] = fig

    return figs


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


def plot_star_proximity_distribution_individual(
    mc: Dict[str, Any],
    R_star_km: float,
    clearance_Rstar: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return star-proximity diagnostics as standalone figures."""
    r_star_all = mc.get("r_star_min", None)
    if r_star_all is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "r_star_min not recorded", ha="center", va="center",
                transform=ax.transAxes)
        return {"star_proximity_distribution_unavailable.png": fig}

    ok = mc["ok"]
    r_ok = r_star_all[ok]
    r_ok = r_ok[np.isfinite(r_ok)]
    r_all_valid = r_star_all[np.isfinite(r_star_all)]

    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(r_all_valid / R_star_km, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_xlabel("r_star_min / R_star", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("All particles - star closest approach", fontsize=12, fontweight="bold")
    if clearance_Rstar is not None:
        ax.axvline(clearance_Rstar, color="red", ls="--", lw=2,
                   label=f"Filter: {clearance_Rstar:.1f} R_star")
        ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["star_proximity_distribution_all_particles.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    if r_ok.size > 0:
        ax.hist(r_ok / R_star_km, bins=40, alpha=0.7, color="darkorange", edgecolor="black")
    ax.set_xlabel("r_star_min / R_star", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Successful particles - star closest approach", fontsize=12, fontweight="bold")
    if clearance_Rstar is not None:
        ax.axvline(clearance_Rstar, color="red", ls="--", lw=2,
                   label=f"Filter: {clearance_Rstar:.1f} R_star")
        ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["star_proximity_distribution_successful_particles.png"] = fig

    return figs


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


def plot_planet_frame_diagnostics_individual(
    analyses: List[Dict[str, Any]],
    R_p_km: float,
    R_star_km: float,
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return planet-frame diagnostics as standalone figures."""
    dv_pf = []
    defl_pf = []
    e_planet = []
    r_min_star = []

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
    vals = np.array(r_min_star) / R_star_km
    idx = np.arange(len(dv_pf))
    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(idx, dv_pf, color="teal", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Delta-v (planet frame) [km/s]")
    ax.set_title("Planet-frame speed change (should ~0 for pure flyby)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["planet_frame_diagnostics_speed_change.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(defl_pf)), defl_pf, color="darkorange", alpha=0.7)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Deflection (deg)")
    ax.set_title("Planet-frame deflection", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["planet_frame_diagnostics_deflection.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(e_planet)), e_planet, color="royalblue", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Delta-E from planet orbit [km^2/s^2]")
    ax.set_title("Energy extracted from planet orbital KE", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["planet_frame_diagnostics_energy_extracted.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(len(vals)), vals, color="crimson", alpha=0.7)
    ax.axhline(1.0, color="gold", ls="--", lw=1.5, label="R_star surface")
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("r_star_min / R_star")
    ax.set_title("Star closest approach per candidate", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["planet_frame_diagnostics_star_closest_approach.png"] = fig

    return figs


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


def plot_parameter_correlations_individual(
    mc: Dict[str, Any],
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return parameter correlation diagnostics as standalone figures."""
    ok = mc["ok"]
    dv = mc["delta_v"][ok]
    defl = mc["deflection"][ok]
    rmin = mc["r_min"][ok]
    dv_vec = mc.get("delta_v_vec", np.full_like(mc["delta_v"], np.nan))[ok]
    r_star = mc.get("r_star_min", np.full_like(mc["delta_v"], np.nan))[ok]
    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(defl, dv, s=12, alpha=0.5, edgecolors="none")
    ax.set_xlabel("Deflection (deg)")
    ax.set_ylabel("Delta-v (km/s)")
    ax.set_title("Delta-v vs Deflection", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["parameter_correlations_delta_v_vs_deflection.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(rmin, dv, s=12, alpha=0.5, c="darkorange", edgecolors="none")
    ax.set_xlabel("r_min (km)")
    ax.set_ylabel("Delta-v (km/s)")
    ax.set_title("Delta-v vs Periapsis", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["parameter_correlations_delta_v_vs_periapsis.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.isfinite(r_star)
    if mask.any():
        ax.scatter(r_star[mask], defl[mask], s=12, alpha=0.5, c="crimson", edgecolors="none")
    ax.set_xlabel("r_star_min (km)")
    ax.set_ylabel("Deflection (deg)")
    ax.set_title("Deflection vs Star proximity", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["parameter_correlations_deflection_vs_star_proximity.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.isfinite(dv_vec)
    if mask.any():
        ax.scatter(dv[mask], dv_vec[mask], s=12, alpha=0.5, c="teal", edgecolors="none")
        lim = max(abs(dv[mask]).max(), abs(dv_vec[mask]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y = x")
        ax.legend(fontsize=9)
    ax.set_xlabel("Scalar Delta-v = |v_f| - |v_i| (km/s)")
    ax.set_ylabel("|Delta-V_vec| = |v_f - v_i| (km/s)")
    ax.set_title("Vector vs Scalar Delta-V", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["parameter_correlations_vector_vs_scalar_delta_v.png"] = fig

    return figs


def plot_energy_cdf(
    mc: Dict[str, Any],
    save_dir: Optional[Path] = None,
    dpi: int = 150,
    *,
    analyses_best: Optional[List[Dict[str, Any]]] = None,
    E_star_narrowed: Optional[float] = None,
    E_planet_narrowed: Optional[float] = None,
    E_3body_best: Optional[float] = None,
    system_name: str = "",
) -> plt.Figure:
    """CDF of ½|ΔV_vec|² for successful particles.

    Parameters
    ----------
    mc : dict
        Monte Carlo results.
    analyses_best : list of dict, optional
        High-res re-run analyses (adds re-run CDF overlay).
    E_star_narrowed : float, optional
        Narrowed star 2-body max ½|ΔV_vec|² (vertical line).
    E_planet_narrowed : float, optional
        Narrowed planet 2-body max ½|ΔV_vec|² (vertical line).
    E_3body_best : float, optional
        Best 3-body ½|ΔV_vec|² from re-runs (vertical line).
    system_name : str
        System name for title.
    """
    ok = mc["ok"]
    # Coarse MC energies — use scalar ΔV as proxy
    energies_mc = 0.5 * mc["delta_v"][ok] ** 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Coarse MC CDF
    if energies_mc.size > 0:
        sorted_e = np.sort(energies_mc)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf, lw=2, color="tab:blue", alpha=0.6,
                label="3-body MC (coarse sweep)")

    # High-res re-run CDF (vector ΔV)
    if analyses_best is not None:
        energies_rerun = np.array([
            a["energy_half_dv_vec_sq"]
            for a in analyses_best if a is not None
        ])
        if len(energies_rerun) > 1:
            sorted_er = np.sort(energies_rerun)
            cdf_r = np.arange(1, len(sorted_er) + 1) / len(sorted_er)
            ax.plot(sorted_er, cdf_r, lw=2, color="tab:purple",
                    label=f"3-body re-run ({len(energies_rerun)} cands, ½|ΔV_vec|²)")

    # Narrowed 2-body baselines
    if E_star_narrowed is not None:
        ax.axvline(E_star_narrowed, color="tab:orange", ls="--", lw=2,
                   label=f"Star max ½|ΔV|² = {E_star_narrowed:.2f}")
    if E_planet_narrowed is not None:
        ax.axvline(E_planet_narrowed, color="tab:green", ls="--", lw=2,
                   label=f"Planet max ½|ΔV|² = {E_planet_narrowed:.2f}")

    # 3-body best
    if E_3body_best is not None:
        ax.axvline(E_3body_best, color="tab:red", ls="-.", lw=2,
                   label=f"3-body best ½|ΔV_vec|² = {E_3body_best:.2f}")

    title = "Scattering Energy CDF — ½|ΔV_vec|² distribution"
    if system_name:
        title = f"Scattering Energy CDF — {system_name}: 2-Body vs 3-Body"
    ax.set_xlabel("Scattering Energy ½|ΔV_vec|²  (km²/s²  ≡  MJ/kg)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "energy_cdf.png", dpi=dpi, bbox_inches="tight")
    return fig


def plot_publication_objectives_dashboard(
    mc: Dict[str, Any],
    analyses_best: Optional[List[Dict[str, Any]]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    R_star_km: Optional[float] = None,
    clearance_Rstar: Optional[float] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Publication-focused diagnostic dashboard (6 panels).

    Panels are designed to provide concrete visual evidence for publication goals:
      1) multi-objective tradeoff and Pareto frontier,
      2) slingshot mechanism (planet-frame changes vs extracted energy),
      3) high-velocity regime coverage,
      4) baseline/ceiling comparison,
      5) deflection distribution,
      6) star-proximity plausibility.
    """
    ok = np.asarray(mc["ok"], dtype=bool)
    dv = np.asarray(mc["delta_v"])[ok]
    dv_vec = np.asarray(mc.get("delta_v_vec", np.full_like(mc["delta_v"], np.nan)))[ok]
    defl = np.asarray(mc["deflection"])[ok]
    e_planet = np.asarray(mc.get("energy_from_planet_orbit", np.full_like(mc["delta_v"], np.nan)))[ok]
    dv_pf = np.asarray(mc.get("delta_v_planet_frame", np.full_like(mc["delta_v"], np.nan)))[ok]
    r_star_min = np.asarray(mc.get("r_star_min", np.full_like(mc["delta_v"], np.nan)))[ok]

    # Keep report layouts to max 2 side-by-side panels.
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # Panel A: Multi-objective tradeoff + Pareto frontier
    ax = axes[0, 0]
    finite_trade = np.isfinite(dv) & np.isfinite(dv_vec)
    if finite_trade.any():
        # Color by extracted energy if available.
        color_mask = finite_trade & np.isfinite(e_planet)
        if color_mask.sum() >= 10:
            sc = ax.scatter(
                dv[color_mask], dv_vec[color_mask], c=e_planet[color_mask],
                s=16, alpha=0.7, cmap="viridis", edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Energy from planet orbit [km²/s²]", fontsize=9)
            if (~color_mask & finite_trade).any():
                ax.scatter(dv[~color_mask & finite_trade], dv_vec[~color_mask & finite_trade],
                           s=12, alpha=0.35, c="gray", edgecolors="none")
        else:
            ax.scatter(dv[finite_trade], dv_vec[finite_trade], s=14, alpha=0.5,
                       c="steelblue", edgecolors="none")

        # Pareto on [dv, dv_vec, e_planet] where possible, else [dv, dv_vec].
        if color_mask.sum() >= 4:
            mat = np.column_stack([dv[color_mask], dv_vec[color_mask], e_planet[color_mask]])
            mask_pf = _pareto_front_mask_max(mat)
            ax.scatter(
                dv[color_mask][mask_pf], dv_vec[color_mask][mask_pf],
                s=48, facecolors="none", edgecolors="crimson", linewidths=1.5,
                label=f"Pareto front (n={mask_pf.sum()})",
            )
        else:
            mat = np.column_stack([dv[finite_trade], dv_vec[finite_trade]])
            mask_pf = _pareto_front_mask_max(mat)
            ax.scatter(
                dv[finite_trade][mask_pf], dv_vec[finite_trade][mask_pf],
                s=48, facecolors="none", edgecolors="crimson", linewidths=1.5,
                label=f"Pareto front (n={mask_pf.sum()})",
            )
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "Insufficient finite points", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Scalar Δv = |v_f|−|v_i| (km/s)")
    ax.set_ylabel("|ΔV_vec| (km/s)")
    ax.set_title("Obj-4: Multi-objective tradeoff")
    ax.grid(True, alpha=0.3)

    # Panel B: Mechanism evidence (planet-frame speed change vs extracted energy)
    ax = axes[0, 1]
    mech_mask = np.isfinite(dv_pf) & np.isfinite(e_planet)
    if mech_mask.any():
        ax.scatter(dv_pf[mech_mask], e_planet[mech_mask], s=18, alpha=0.7,
                   c="darkorange", edgecolors="none")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axhline(0.0, color="gray", lw=0.8)
        if mech_mask.sum() >= 3:
            corr = np.corrcoef(dv_pf[mech_mask], e_planet[mech_mask])[0, 1]
            ax.text(0.02, 0.98, f"corr={corr:.2f}", transform=ax.transAxes,
                    va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No planet-frame diagnostic data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Δv (planet frame) [km/s]")
    ax.set_ylabel("Energy from planet orbit [km²/s²]")
    ax.set_title("Obj-1: Slingshot mechanism evidence")
    ax.grid(True, alpha=0.3)

    # Panel C: High-velocity coverage (initial speed vs achieved scalar Δv)
    ax = axes[1, 0]
    sat = mc.get("sat_states", None)
    if sat is not None and len(sat) == len(mc["ok"]):
        v_init = np.hypot(sat[:, 2], sat[:, 3])
        v_init_ok = v_init[ok]
        cov_mask = np.isfinite(v_init_ok) & np.isfinite(dv)
        if cov_mask.any():
            ax.scatter(v_init_ok[cov_mask], dv[cov_mask], s=16, alpha=0.6,
                       c="tab:green", edgecolors="none")
            ax.text(0.02, 0.98,
                    f"v_init range: [{v_init_ok[cov_mask].min():.1f}, {v_init_ok[cov_mask].max():.1f}] km/s",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No finite coverage points", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "sat_states unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Initial barycentric speed (km/s)")
    ax.set_ylabel("Scalar Δv (km/s)")
    ax.set_title("Obj-2: High-velocity regime coverage")
    ax.grid(True, alpha=0.3)

    # Panel D: Analytic/baseline ceiling comparison
    ax = axes[1, 1]
    labels = []
    vals = []
    if comparison is not None:
        if comparison.get("dv_vec_2body_star") is not None:
            labels.append("2-body star")
            vals.append(0.5 * float(comparison["dv_vec_2body_star"]) ** 2)
        if comparison.get("dv_vec_2body_planet") is not None:
            labels.append("2-body planet")
            vals.append(0.5 * float(comparison["dv_vec_2body_planet"]) ** 2)
        if comparison.get("dv_vec_3body") is not None:
            labels.append("3-body best")
            vals.append(0.5 * float(comparison["dv_vec_3body"]) ** 2)
    if labels:
        bars = ax.bar(labels, vals, color=["#f4a261", "#2a9d8f", "#457b9d"][:len(labels)], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Baseline comparison unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("0.5|ΔV_vec|² (km²/s²)")
    ax.set_title("Obj-3: Ceiling / baseline comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel E: Deflection distribution
    ax = axes[2, 0]
    finite_defl = np.isfinite(defl)
    if finite_defl.any():
        ax.hist(defl[finite_defl], bins=24, alpha=0.75, color="mediumpurple", edgecolor="black")
        ax.axvline(0.0, color="gray", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No deflection data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Deflection (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Directional redistribution (SETI-relevant)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel F: Star proximity vs Δv plausibility
    ax = axes[2, 1]
    prox_mask = np.isfinite(r_star_min) & np.isfinite(dv)
    if prox_mask.any():
        if R_star_km and R_star_km > 0:
            x = r_star_min[prox_mask] / R_star_km
            ax.set_xlabel("r_star_min / R★")
            if clearance_Rstar is not None:
                ax.axvline(clearance_Rstar, color="red", ls="--", lw=1.3,
                           label=f"Filter {clearance_Rstar:.1f} R★")
                ax.legend(fontsize=8)
        else:
            x = r_star_min[prox_mask]
            ax.set_xlabel("r_star_min (km)")
        ax.scatter(x, dv[prox_mask], s=16, alpha=0.65, c="crimson", edgecolors="none")
    else:
        ax.text(0.5, 0.5, "No star-proximity data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("Scalar Δv (km/s)")
    ax.set_title("Physical plausibility vs stellar proximity")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Publication Objectives Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "publication_objectives_dashboard.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_publication_objectives_individual(
    mc: Dict[str, Any],
    analyses_best: Optional[List[Dict[str, Any]]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    R_star_km: Optional[float] = None,
    clearance_Rstar: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return publication-objective diagnostics as standalone figures."""
    _ = analyses_best
    ok = np.asarray(mc["ok"], dtype=bool)
    dv = np.asarray(mc["delta_v"])[ok]
    dv_vec = np.asarray(mc.get("delta_v_vec", np.full_like(mc["delta_v"], np.nan)))[ok]
    defl = np.asarray(mc["deflection"])[ok]
    e_planet = np.asarray(mc.get("energy_from_planet_orbit", np.full_like(mc["delta_v"], np.nan)))[ok]
    dv_pf = np.asarray(mc.get("delta_v_planet_frame", np.full_like(mc["delta_v"], np.nan)))[ok]
    r_star_min = np.asarray(mc.get("r_star_min", np.full_like(mc["delta_v"], np.nan)))[ok]
    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    finite_trade = np.isfinite(dv) & np.isfinite(dv_vec)
    if finite_trade.any():
        color_mask = finite_trade & np.isfinite(e_planet)
        if color_mask.sum() >= 10:
            sc = ax.scatter(
                dv[color_mask], dv_vec[color_mask], c=e_planet[color_mask],
                s=16, alpha=0.7, cmap="viridis", edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Energy from planet orbit [km^2/s^2]", fontsize=9)
            if (~color_mask & finite_trade).any():
                ax.scatter(dv[~color_mask & finite_trade], dv_vec[~color_mask & finite_trade],
                           s=12, alpha=0.35, c="gray", edgecolors="none")
        else:
            ax.scatter(dv[finite_trade], dv_vec[finite_trade], s=14, alpha=0.5,
                       c="steelblue", edgecolors="none")
        if color_mask.sum() >= 4:
            mat = np.column_stack([dv[color_mask], dv_vec[color_mask], e_planet[color_mask]])
            mask_pf = _pareto_front_mask_max(mat)
            ax.scatter(
                dv[color_mask][mask_pf], dv_vec[color_mask][mask_pf],
                s=48, facecolors="none", edgecolors="crimson", linewidths=1.5,
                label=f"Pareto front (n={mask_pf.sum()})",
            )
        else:
            mat = np.column_stack([dv[finite_trade], dv_vec[finite_trade]])
            mask_pf = _pareto_front_mask_max(mat)
            ax.scatter(
                dv[finite_trade][mask_pf], dv_vec[finite_trade][mask_pf],
                s=48, facecolors="none", edgecolors="crimson", linewidths=1.5,
                label=f"Pareto front (n={mask_pf.sum()})",
            )
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "Insufficient finite points", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Scalar Delta-v = |v_f|-|v_i| (km/s)")
    ax.set_ylabel("|Delta-V_vec| (km/s)")
    ax.set_title("Obj-4: Multi-objective tradeoff", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["publication_objectives_obj4_tradeoff.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    mech_mask = np.isfinite(dv_pf) & np.isfinite(e_planet)
    if mech_mask.any():
        ax.scatter(dv_pf[mech_mask], e_planet[mech_mask], s=18, alpha=0.7,
                   c="darkorange", edgecolors="none")
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axhline(0.0, color="gray", lw=0.8)
        if mech_mask.sum() >= 3:
            corr = np.corrcoef(dv_pf[mech_mask], e_planet[mech_mask])[0, 1]
            ax.text(0.02, 0.98, f"corr={corr:.2f}", transform=ax.transAxes,
                    va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No planet-frame diagnostic data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Delta-v (planet frame) [km/s]")
    ax.set_ylabel("Energy from planet orbit [km^2/s^2]")
    ax.set_title("Obj-1: Slingshot mechanism evidence", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["publication_objectives_obj1_mechanism.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    sat = mc.get("sat_states", None)
    if sat is not None and len(sat) == len(mc["ok"]):
        v_init = np.hypot(sat[:, 2], sat[:, 3])
        v_init_ok = v_init[ok]
        cov_mask = np.isfinite(v_init_ok) & np.isfinite(dv)
        if cov_mask.any():
            ax.scatter(v_init_ok[cov_mask], dv[cov_mask], s=16, alpha=0.6,
                       c="tab:green", edgecolors="none")
            ax.text(0.02, 0.98,
                    f"v_init range: [{v_init_ok[cov_mask].min():.1f}, {v_init_ok[cov_mask].max():.1f}] km/s",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No finite coverage points", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "sat_states unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Initial barycentric speed (km/s)")
    ax.set_ylabel("Scalar Delta-v (km/s)")
    ax.set_title("Obj-2: High-velocity regime coverage", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["publication_objectives_obj2_high_velocity_coverage.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    labels = []
    vals = []
    if comparison is not None:
        if comparison.get("dv_vec_2body_star") is not None:
            labels.append("2-body star")
            vals.append(0.5 * float(comparison["dv_vec_2body_star"]) ** 2)
        if comparison.get("dv_vec_2body_planet") is not None:
            labels.append("2-body planet")
            vals.append(0.5 * float(comparison["dv_vec_2body_planet"]) ** 2)
        if comparison.get("dv_vec_3body") is not None:
            labels.append("3-body best")
            vals.append(0.5 * float(comparison["dv_vec_3body"]) ** 2)
    if labels:
        bars = ax.bar(labels, vals, color=["#f4a261", "#2a9d8f", "#457b9d"][:len(labels)], alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Baseline comparison unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("0.5|Delta-V_vec|^2 (km^2/s^2)")
    ax.set_title("Obj-3: Ceiling / baseline comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["publication_objectives_obj3_baseline_ceiling.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    finite_defl = np.isfinite(defl)
    if finite_defl.any():
        ax.hist(defl[finite_defl], bins=24, alpha=0.75, color="mediumpurple", edgecolor="black")
        ax.axvline(0.0, color="gray", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No deflection data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Deflection (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Directional redistribution (SETI-relevant)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["publication_objectives_obj5_deflection_distribution.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    prox_mask = np.isfinite(r_star_min) & np.isfinite(dv)
    if prox_mask.any():
        if R_star_km and R_star_km > 0:
            x = r_star_min[prox_mask] / R_star_km
            ax.set_xlabel("r_star_min / R_star")
            if clearance_Rstar is not None:
                ax.axvline(clearance_Rstar, color="red", ls="--", lw=1.3,
                           label=f"Filter {clearance_Rstar:.1f} R_star")
                ax.legend(fontsize=8)
        else:
            x = r_star_min[prox_mask]
            ax.set_xlabel("r_star_min (km)")
        ax.scatter(x, dv[prox_mask], s=16, alpha=0.65, c="crimson", edgecolors="none")
    else:
        ax.text(0.5, 0.5, "No star-proximity data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylabel("Scalar Delta-v (km/s)")
    ax.set_title("Physical plausibility vs stellar proximity", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["publication_objectives_obj6_stellar_proximity.png"] = fig

    return figs


def plot_candidate_ranking_diagnostics(
    analyses: List[Optional[Dict[str, Any]]],
    top_indices: Optional[np.ndarray] = None,
    R_p_km: Optional[float] = None,
    R_star_km: Optional[float] = None,
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Detailed ranking diagnostics for re-run candidates."""
    rows = []
    for i, ana in enumerate(analyses):
        if ana is None:
            continue
        enc = ana.get("encounter")
        mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
        rmin = ana.get("r_min", np.nan)
        rstar = _enc_field(enc, "r_star_min", np.nan)
        rows.append({
            "mc_idx": mc_idx,
            "delta_v": float(ana.get("delta_v", np.nan)),
            "delta_v_vec": float(ana.get("delta_v_vec", np.nan)),
            "deflection": float(ana.get("deflection", np.nan)),
            "dv_pf": float(ana.get("delta_v_planet_frame", np.nan)),
            "e_orbit": float(ana.get("energy_from_planet_orbit", np.nan)),
            "rmin_rp": float(rmin / R_p_km) if R_p_km and np.isfinite(rmin) else np.nan,
            "rstar_rs": float(rstar / R_star_km) if R_star_km and np.isfinite(rstar) else np.nan,
        })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if not rows:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No valid re-run candidates", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        return fig

    rows = sorted(rows, key=lambda r: r["delta_v"], reverse=True)
    labels = [f"MC#{r['mc_idx']}" for r in rows]
    x = np.arange(len(rows))

    # A) scalar & vector speed-change bars
    ax = axes[0, 0]
    w = 0.38
    ax.bar(x - w / 2, [r["delta_v"] for r in rows], width=w, alpha=0.8, color="tab:blue", label="Δv (scalar)")
    ax.bar(x + w / 2, [r["delta_v_vec"] for r in rows], width=w, alpha=0.8, color="tab:orange", label="|ΔV_vec|")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("km/s")
    ax.set_title("Candidate speed-change ranking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # B) deflection bars
    ax = axes[0, 1]
    ax.bar(x, [r["deflection"] for r in rows], alpha=0.8, color="tab:purple")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("deg")
    ax.set_title("Deflection ranking")
    ax.grid(True, alpha=0.3, axis="y")

    # C) mechanism plane
    ax = axes[1, 0]
    dvpf = np.array([r["dv_pf"] for r in rows], dtype=float)
    eorb = np.array([r["e_orbit"] for r in rows], dtype=float)
    mask = np.isfinite(dvpf) & np.isfinite(eorb)
    if mask.any():
        ax.scatter(dvpf[mask], eorb[mask], s=45, alpha=0.85, c="tab:green")
        for xi, yi, lbl in zip(dvpf[mask], eorb[mask], np.array(labels)[mask]):
            ax.text(xi, yi, lbl, fontsize=7, alpha=0.85)
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axhline(0.0, color="gray", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No finite mechanism diagnostics", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Δv planet frame (km/s)")
    ax.set_ylabel("Energy from planet orbit (km²/s²)")
    ax.set_title("Mechanism diagnostics by candidate")
    ax.grid(True, alpha=0.3)

    # D) closest-approach in body radii
    ax = axes[1, 1]
    rmin_rp = np.array([r["rmin_rp"] for r in rows], dtype=float)
    rstar_rs = np.array([r["rstar_rs"] for r in rows], dtype=float)
    ax.plot(x, rmin_rp, "o-", label="r_min / R_p", lw=1.5, markersize=5)
    if np.isfinite(rstar_rs).any():
        ax.plot(x, rstar_rs, "s--", label="r_star_min / R★", lw=1.5, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Body radii")
    ax.set_title("Closest-approach diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Candidate Ranking Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "candidate_ranking_diagnostics.png",
                    dpi=dpi, bbox_inches="tight")
    return fig


def plot_candidate_ranking_diagnostics_individual(
    analyses: List[Optional[Dict[str, Any]]],
    top_indices: Optional[np.ndarray] = None,
    R_p_km: Optional[float] = None,
    R_star_km: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Return candidate-ranking diagnostics as standalone figures."""
    rows = []
    for i, ana in enumerate(analyses):
        if ana is None:
            continue
        enc = ana.get("encounter")
        mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
        rmin = ana.get("r_min", np.nan)
        rstar = _enc_field(enc, "r_star_min", np.nan)
        rows.append({
            "mc_idx": mc_idx,
            "delta_v": float(ana.get("delta_v", np.nan)),
            "delta_v_vec": float(ana.get("delta_v_vec", np.nan)),
            "deflection": float(ana.get("deflection", np.nan)),
            "dv_pf": float(ana.get("delta_v_planet_frame", np.nan)),
            "e_orbit": float(ana.get("energy_from_planet_orbit", np.nan)),
            "rmin_rp": float(rmin / R_p_km) if R_p_km and np.isfinite(rmin) else np.nan,
            "rstar_rs": float(rstar / R_star_km) if R_star_km and np.isfinite(rstar) else np.nan,
        })

    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid re-run candidates", ha="center", va="center",
                transform=ax.transAxes)
        return {"candidate_ranking_diagnostics_unavailable.png": fig}

    rows = sorted(rows, key=lambda r: r["delta_v"], reverse=True)
    labels = [f"MC#{r['mc_idx']}" for r in rows]
    x = np.arange(len(rows))
    figs: Dict[str, plt.Figure] = {}

    fig, ax = plt.subplots(figsize=figsize)
    w = 0.38
    ax.bar(x - w / 2, [r["delta_v"] for r in rows], width=w, alpha=0.8, color="tab:blue", label="Delta-v (scalar)")
    ax.bar(x + w / 2, [r["delta_v_vec"] for r in rows], width=w, alpha=0.8, color="tab:orange", label="|Delta-V_vec|")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("km/s")
    ax.set_title("Candidate speed-change ranking", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["candidate_ranking_speed_change.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, [r["deflection"] for r in rows], alpha=0.8, color="tab:purple")
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("deg")
    ax.set_title("Deflection ranking", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs["candidate_ranking_deflection.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    dvpf = np.array([r["dv_pf"] for r in rows], dtype=float)
    eorb = np.array([r["e_orbit"] for r in rows], dtype=float)
    mask = np.isfinite(dvpf) & np.isfinite(eorb)
    if mask.any():
        ax.scatter(dvpf[mask], eorb[mask], s=45, alpha=0.85, c="tab:green")
        for xi, yi, lbl in zip(dvpf[mask], eorb[mask], np.array(labels)[mask]):
            ax.text(xi, yi, lbl, fontsize=7, alpha=0.85)
        ax.axvline(0.0, color="gray", lw=0.8)
        ax.axhline(0.0, color="gray", lw=0.8)
    else:
        ax.text(0.5, 0.5, "No finite mechanism diagnostics", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Delta-v planet frame (km/s)")
    ax.set_ylabel("Energy from planet orbit (km^2/s^2)")
    ax.set_title("Mechanism diagnostics by candidate", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs["candidate_ranking_mechanism_plane.png"] = fig

    fig, ax = plt.subplots(figsize=figsize)
    rmin_rp = np.array([r["rmin_rp"] for r in rows], dtype=float)
    rstar_rs = np.array([r["rstar_rs"] for r in rows], dtype=float)
    ax.plot(x, rmin_rp, "o-", label="r_min / R_p", lw=1.5, markersize=5)
    if np.isfinite(rstar_rs).any():
        ax.plot(x, rstar_rs, "s--", label="r_star_min / R_star", lw=1.5, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Body radii")
    ax.set_title("Closest-approach diagnostics", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs["candidate_ranking_closest_approach.png"] = fig

    return figs
