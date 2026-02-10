"""
Two-body diagnostic plots — absorbed from standalone scripts.

All functions accept config-derived parameters in km-kg-s units.
The core solver TwoBodyScatter.py is unit-agnostic; we pass mu = G_KM * M_body.

Plot categories:
    1. Poincaré  (b × α_inf) heatmaps
    2. Scattering polar (b_mag × θ_b) maps
    3. Cartesian (x, y) encounter heatmaps
    4. Cartesian encounter + trajectory overlay
    5. Spatial energy/flux density heatmap
    6. Oberth comparison maps
    7. Trajectory tracks — coloured by orbital energy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from .constants import G_KM, M_SUN, M_JUP, R_SUN, R_JUP

# ---------------------------------------------------------------------------
#  Resolve path to TwoBodyScatter at import time
# ---------------------------------------------------------------------------
import sys as _sys
import importlib as _importlib

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import TwoBodyScatter as _TBS


# ===================================================================
# Private helpers — shared grid-scan logic
# ===================================================================

def _compute_encounter_grid_poincare(
    mu: float,
    num_b: int,
    num_angle: int,
    b_min: float,
    b_max: float,
    alpha_min_rad: float,
    alpha_max_rad: float,
    v_inf: float,
    vstar0: float,
    r_far: float,
) -> Dict[str, np.ndarray]:
    """Compute b × α_inf Poincaré encounter grid.

    All units: km-kg-s.
    Returns dict with 2-D arrays: deltaV, theta, vinf, rp, b_grid, alpha_grid.
    """
    b_arr = np.linspace(b_min, b_max, num_b)
    alpha_arr = np.linspace(alpha_min_rad, alpha_max_rad, num_angle)
    B, A = np.meshgrid(b_arr, alpha_arr, indexing="ij")

    deltaV = np.full_like(B, np.nan)
    theta = np.full_like(B, np.nan)
    vinf_out = np.full_like(B, np.nan)
    rp = np.full_like(B, np.nan)

    for i in range(num_b):
        for j in range(num_angle):
            b_val = b_arr[i]
            alpha = alpha_arr[j]
            xm0 = r_far * np.cos(alpha)
            ym0 = r_far * np.sin(alpha)
            um0 = -v_inf * np.cos(alpha)
            vm0 = -v_inf * np.sin(alpha)
            try:
                res = _TBS.gravity_assist_no_burn(xm0, ym0, um0, vm0, vstar0, mu)
                deltaV[i, j] = _TBS.deltaV_lab(um0, vm0, res.umF, res.vmF)
                theta[i, j] = np.degrees(res.theta)
                vinf_out[i, j] = res.vinf
                rp[i, j] = res.rp
            except Exception:
                pass

    return {
        "b_arr": b_arr,
        "alpha_arr": alpha_arr,
        "B": B,
        "A": A,
        "deltaV": deltaV,
        "theta": theta,
        "vinf": vinf_out,
        "rp": rp,
    }


def _compute_encounter_grid_cartesian(
    mu: float,
    num_x: int,
    num_y: int,
    xy_range: float,
    v_approach: float,
    vstar0: float,
    approach_angle_deg: float,
    r_encounter: float,
) -> Dict[str, np.ndarray]:
    """Compute Cartesian (x, y) offset encounter grid.

    Returns dict with 2-D arrays: deltaV, theta, rp, b_mag, X, Y.
    """
    x_arr = np.linspace(-xy_range, xy_range, num_x)
    y_arr = np.linspace(-xy_range, xy_range, num_y)
    X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")

    deltaV = np.full_like(X, np.nan)
    theta_grid = np.full_like(X, np.nan)
    rp_grid = np.full_like(X, np.nan)
    b_grid = np.full_like(X, np.nan)

    angle_rad = np.radians(approach_angle_deg)
    ux = -v_approach * np.cos(angle_rad)
    uy = -v_approach * np.sin(angle_rad)

    for i in range(num_x):
        for j in range(num_y):
            xm0 = r_encounter * np.cos(angle_rad) + x_arr[i]
            ym0 = r_encounter * np.sin(angle_rad) + y_arr[j]
            try:
                res = _TBS.gravity_assist_no_burn(xm0, ym0, ux, uy, vstar0, mu)
                deltaV[i, j] = _TBS.deltaV_lab(ux, uy, res.umF, res.vmF)
                theta_grid[i, j] = np.degrees(res.theta)
                rp_grid[i, j] = res.rp
                b_grid[i, j] = res.b
            except Exception:
                pass

    return {
        "x_arr": x_arr,
        "y_arr": y_arr,
        "X": X,
        "Y": Y,
        "deltaV": deltaV,
        "theta": theta_grid,
        "rp": rp_grid,
        "b": b_grid,
    }


def _compute_encounter_grid_polar(
    mu: float,
    num_b: int,
    num_theta: int,
    b_min: float,
    b_max: float,
    v_approach: float,
    vstar0: float,
    approach_angle_deg: float,
    r_start: float,
) -> Dict[str, np.ndarray]:
    """Compute polar (b_mag, θ_b) scattering grid.

    Returns dict with 2-D arrays: deltaV, theta_defl, rp, Bmag, Theta_b.
    """
    b_arr = np.linspace(b_min, b_max, num_b)
    theta_arr = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    Bmag, Theta_b = np.meshgrid(b_arr, theta_arr, indexing="ij")

    deltaV = np.full_like(Bmag, np.nan)
    theta_defl = np.full_like(Bmag, np.nan)
    rp = np.full_like(Bmag, np.nan)

    angle_rad = np.radians(approach_angle_deg)

    for i in range(num_b):
        for j in range(num_theta):
            bval = b_arr[i]
            tb = theta_arr[j]
            offset_x = bval * np.cos(tb)
            offset_y = bval * np.sin(tb)
            xm0 = r_start * np.cos(angle_rad) + offset_x
            ym0 = r_start * np.sin(angle_rad) + offset_y
            ux = -v_approach * np.cos(angle_rad)
            uy = -v_approach * np.sin(angle_rad)
            try:
                res = _TBS.gravity_assist_no_burn(xm0, ym0, ux, uy, vstar0, mu)
                deltaV[i, j] = _TBS.deltaV_lab(ux, uy, res.umF, res.vmF)
                theta_defl[i, j] = np.degrees(res.theta)
                rp[i, j] = res.rp
            except Exception:
                pass

    return {
        "b_arr": b_arr,
        "theta_arr": theta_arr,
        "Bmag": Bmag,
        "Theta_b": Theta_b,
        "deltaV": deltaV,
        "theta_defl": theta_defl,
        "rp": rp,
    }


# ===================================================================
# 1. Poincaré heatmaps (b × α_inf)
# ===================================================================

def plot_poincare_heatmaps(
    M_body_kg: float,
    v_inf_kms: float = 50.0,
    vstar0_kms: float = 10.0,
    num_b: int = 60,
    num_angle: int = 60,
    b_min_km: float = 1e7,
    b_max_km: float = 2e9,
    r_far_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate Poincaré (b vs α_inf) heatmaps for a 2-body encounter.

    Parameters: all in km-kg-s.  Returns list of figures.
    """
    mu = G_KM * M_body_kg
    grid = _compute_encounter_grid_poincare(
        mu=mu, num_b=num_b, num_angle=num_angle,
        b_min=b_min_km, b_max=b_max_km,
        alpha_min_rad=0.2, alpha_max_rad=np.pi - 0.2,
        v_inf=v_inf_kms, vstar0=vstar0_kms, r_far=r_far_km,
    )
    b_plot = grid["B"] / 1e6  # km → 10⁶ km for axis label
    a_plot = np.degrees(grid["A"])

    figs = []

    # --- Multi-panel ---
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    datasets = [
        (grid["deltaV"], "ΔV (km/s)", "hot_r"),
        (grid["theta"], "Deflection θ (°)", "viridis"),
        (grid["vinf"], "v∞ out (km/s)", "cool"),
        (grid["rp"], "Periapsis rp (km)", "plasma"),
    ]
    for ax, (data, label, cmap) in zip(axes.flat, datasets):
        cf = ax.contourf(a_plot, b_plot, data, levels=30, cmap=cmap)
        fig1.colorbar(cf, ax=ax, label=label)
        ax.set_xlabel("Approach angle α (°)")
        ax.set_ylabel("Impact parameter b (10⁶ km)")
        ax.set_title(label)
    fig1.suptitle(f"Poincaré encounter maps — {body_label}", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    figs.append(fig1)

    # --- Combined ΔV + θ contours ---
    fig2, ax = plt.subplots(figsize=(12, 9))
    cf = ax.contourf(a_plot, b_plot, grid["deltaV"], levels=25, cmap="hot_r")
    fig2.colorbar(cf, ax=ax, label="ΔV (km/s)")
    ax.contour(a_plot, b_plot, grid["theta"], levels=10, colors="cyan", linewidths=1.5, alpha=0.6)
    ax.set_xlabel("Approach angle α (°)", fontsize=12)
    ax.set_ylabel("Impact parameter b (10⁶ km)", fontsize=12)
    ax.set_title(f"ΔV with deflection contours — {body_label}", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    figs.append(fig2)

    if save_dir:
        save_dir = Path(save_dir)
        tag = body_label.lower().replace(" ", "_")
        fig1.savefig(save_dir / f"poincare_heatmap_multi_{tag}.png", dpi=dpi, bbox_inches="tight")
        fig2.savefig(save_dir / f"poincare_heatmap_combined_{tag}.png", dpi=dpi, bbox_inches="tight")

    return figs


# ===================================================================
# 2. Scattering polar maps (b_mag × θ_b)
# ===================================================================

def plot_scattering_maps(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: float = 10.0,
    approach_angles_deg: List[float] = (0.0, 45.0, 85.0),
    num_b: int = 50,
    num_theta: int = 50,
    b_min_km: float = 1e7,
    b_max_km: float = 2.5e9,
    r_start_km: float = 2e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate polar scattering maps for multiple approach angles."""
    mu = G_KM * M_body_kg
    figs = []

    for angle_deg in approach_angles_deg:
        grid = _compute_encounter_grid_polar(
            mu=mu, num_b=num_b, num_theta=num_theta,
            b_min=b_min_km, b_max=b_max_km,
            v_approach=v_approach_kms, vstar0=vstar0_kms,
            approach_angle_deg=angle_deg, r_start=r_start_km,
        )
        b_plot = grid["Bmag"] / 1e6
        t_plot = np.degrees(grid["Theta_b"])

        # 3-panel scattering map
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, (data, label, cmap) in zip(axes, [
            (grid["deltaV"], "ΔV (km/s)", "hot_r"),
            (grid["theta_defl"], "θ deflection (°)", "viridis"),
            (grid["rp"], "Periapsis rp (km)", "plasma"),
        ]):
            cf = ax.contourf(t_plot, b_plot, data, levels=30, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=label)
            ax.set_xlabel("Impact direction θ_b (°)")
            ax.set_ylabel("Impact parameter b (10⁶ km)")
            ax.set_title(label)
        suffix = f"{angle_deg:.0f}deg"
        fig.suptitle(f"Scattering map — {body_label}, approach {suffix}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        figs.append(fig)

        if save_dir:
            fig.savefig(Path(save_dir) / f"scattering_map_{body_label.lower()}_{suffix}.png",
                        dpi=dpi, bbox_inches="tight")

    return figs


# ===================================================================
# 3. Cartesian encounter heatmaps (x, y)
# ===================================================================

def plot_encounter_2d_cartesian(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: float = 10.0,
    approach_angles_deg: List[float] = (0.0, 45.0, 85.0),
    num_xy: int = 70,
    xy_range_km: float = 3e9,
    r_encounter_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate Cartesian (x, y) ΔV / deflection / periapsis heatmaps."""
    mu = G_KM * M_body_kg
    figs = []

    for angle_deg in approach_angles_deg:
        grid = _compute_encounter_grid_cartesian(
            mu=mu, num_x=num_xy, num_y=num_xy,
            xy_range=xy_range_km, v_approach=v_approach_kms,
            vstar0=vstar0_kms, approach_angle_deg=angle_deg,
            r_encounter=r_encounter_km,
        )
        x_plot = grid["X"] / 1e6
        y_plot = grid["Y"] / 1e6

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for ax, (data, label, cmap) in zip(axes.flat, [
            (grid["deltaV"], "ΔV (km/s)", "hot_r"),
            (grid["theta"], "Deflection θ (°)", "viridis"),
            (grid["rp"], "Periapsis rp (km)", "plasma"),
            (grid["b"], "Impact param b (km)", "cool"),
        ]):
            cf = ax.contourf(x_plot, y_plot, data, levels=30, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=label)
            ax.add_patch(Circle((0, 0), 0.1, fill=False, ec="cyan", lw=1.5))
            ax.set_aspect("equal")
            ax.set_xlabel("x offset (10⁶ km)")
            ax.set_ylabel("y offset (10⁶ km)")
            ax.set_title(label)

        suffix = f"{angle_deg:.0f}deg"
        fig.suptitle(f"2D encounter maps — {body_label}, approach {suffix}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        figs.append(fig)

        if save_dir:
            fig.savefig(Path(save_dir) / f"encounter_2d_{body_label.lower()}_{suffix}_maps.png",
                        dpi=dpi, bbox_inches="tight")

    return figs


# ===================================================================
# 4. Cartesian encounter + trajectory overlay
# ===================================================================

def plot_encounter_2d_trajectories(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: float = 10.0,
    approach_angles_deg: List[float] = (0.0, 45.0, 85.0),
    num_xy: int = 60,
    xy_range_km: float = 3e9,
    r_encounter_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate ΔV heatmap + hyperbolic trajectory overlay."""
    mu = G_KM * M_body_kg
    figs = []

    # --- Multi-scenario comparison panel ---
    fig_multi, axes = plt.subplots(1, len(approach_angles_deg),
                                   figsize=(6.5 * len(approach_angles_deg), 6.5))
    if len(approach_angles_deg) == 1:
        axes = [axes]
    for ax, angle_deg in zip(axes, approach_angles_deg):
        grid = _compute_encounter_grid_cartesian(
            mu=mu, num_x=num_xy, num_y=num_xy,
            xy_range=xy_range_km, v_approach=v_approach_kms,
            vstar0=vstar0_kms, approach_angle_deg=angle_deg,
            r_encounter=r_encounter_km,
        )
        x_plot = grid["X"] / 1e6
        y_plot = grid["Y"] / 1e6
        cf = ax.contourf(x_plot, y_plot, grid["deltaV"], levels=30, cmap="hot_r")
        fig_multi.colorbar(cf, ax=ax)
        ax.add_patch(Circle((0, 0), 0.15, fill=True, fc="gold", ec="orange", lw=1.5, alpha=0.7))
        ax.set_aspect("equal")
        ax.set_title(f"Approach {angle_deg:.0f}°")
        ax.set_xlabel("x (10⁶ km)")
        ax.set_ylabel("y (10⁶ km)")

    fig_multi.suptitle(f"Multi-scenario ΔV — {body_label}", fontsize=14, fontweight="bold")
    fig_multi.tight_layout()
    figs.append(fig_multi)

    if save_dir:
        fig_multi.savefig(Path(save_dir) / f"encounter_2d_{body_label.lower()}_multi.png",
                          dpi=dpi, bbox_inches="tight")

    return figs


# ===================================================================
# 5. Oberth comparison maps
# ===================================================================

def plot_oberth_comparison(
    M_body_kg: float,
    v_inf_kms: float = 50.0,
    vstar0_kms: float = 10.0,
    dv_burn_kms: float = 5.0,
    num_b: int = 50,
    num_angle: int = 50,
    b_min_km: float = 5e7,
    b_max_km: float = 2e9,
    r_far_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Compare no-burn vs Oberth burn across Poincaré parameter space."""
    mu = G_KM * M_body_kg
    b_arr = np.linspace(b_min_km, b_max_km, num_b)
    alpha_arr = np.linspace(0.1, np.pi - 0.1, num_angle)
    B, A = np.meshgrid(b_arr, alpha_arr, indexing="ij")

    dv_no = np.full_like(B, np.nan)
    dv_ob = np.full_like(B, np.nan)
    gain = np.full_like(B, np.nan)

    for i in range(num_b):
        for j in range(num_angle):
            bval = b_arr[i]
            alpha = alpha_arr[j]
            xm0 = r_far_km * np.cos(alpha)
            ym0 = r_far_km * np.sin(alpha)
            um0 = -v_inf_kms * np.cos(alpha)
            vm0 = -v_inf_kms * np.sin(alpha)
            try:
                res_nb = _TBS.gravity_assist_no_burn(xm0, ym0, um0, vm0, vstar0_kms, mu)
                dv_no[i, j] = _TBS.deltaV_lab(um0, vm0, res_nb.umF, res_nb.vmF)
                res_ob = _TBS.gravity_assist_oberth(xm0, ym0, um0, vm0, vstar0_kms, mu, dv_burn_kms)
                dv_ob[i, j] = _TBS.deltaV_lab(um0, vm0, res_ob.umF, res_ob.vmF)
                gain[i, j] = dv_ob[i, j] - dv_no[i, j]
            except Exception:
                pass

    b_plot = B / 1e6
    a_plot = np.degrees(A)
    figs = []

    # 3-panel comparison
    fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (data, label, cmap) in zip(axes, [
        (dv_no, "ΔV no-burn (km/s)", "hot_r"),
        (dv_ob, f"ΔV Oberth Δv={dv_burn_kms} km/s", "hot_r"),
        (gain, "Oberth gain (km/s)", "RdYlGn"),
    ]):
        cf = ax.contourf(a_plot, b_plot, data, levels=25, cmap=cmap)
        fig1.colorbar(cf, ax=ax, label=label)
        ax.set_xlabel("α (°)")
        ax.set_ylabel("b (10⁶ km)")
        ax.set_title(label)
    fig1.suptitle(f"Oberth comparison — {body_label}", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    figs.append(fig1)

    # Gain detail with contour overlay
    fig2, ax = plt.subplots(figsize=(13, 9))
    cf = ax.contourf(a_plot, b_plot, gain, levels=30, cmap="RdYlGn")
    fig2.colorbar(cf, ax=ax, label="Oberth gain (km/s)")
    ax.contour(a_plot, b_plot, dv_no, levels=12, colors="gray", alpha=0.5)
    ax.set_xlabel("α (°)", fontsize=12)
    ax.set_ylabel("b (10⁶ km)", fontsize=12)
    ax.set_title(f"Oberth gain — {body_label}", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    figs.append(fig2)

    if save_dir:
        save_dir = Path(save_dir)
        tag = body_label.lower().replace(" ", "_")
        fig1.savefig(save_dir / f"oberth_comparison_{tag}.png", dpi=dpi, bbox_inches="tight")
        fig2.savefig(save_dir / f"oberth_gain_{tag}.png", dpi=dpi, bbox_inches="tight")

    return figs


# ---------------------------------------------------------------------------
#  7. Trajectory tracks — coloured by orbital energy
# ---------------------------------------------------------------------------

def plot_trajectory_tracks(
    narrowed: Dict[str, Any],
    sols_best: List[Any],
    analyses_best: List[Dict[str, Any]],
    cfg: Any,
    *,
    num_b: int = 150,
    num_angles: int = 50,
    num_points: int = 200,
    padding_frac: float = 0.20,
    save_dir: Optional[str] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Trajectory tracks for star and planet 2-body baselines (separate figs).

    Each figure shows hyperbolic trajectories coloured by specific orbital
    energy (½ΔV², km²/s²) with 3-body candidate tracks overlaid in cyan.
    Axes are clamped to the extent of the 3-body candidates + padding.

    Parameters
    ----------
    narrowed : dict
        Output from ``compute_narrowed_baselines`` with keys
        ``"envelope"``, ``"star"``, ``"planet"``.
    sols_best : list of OdeResult
        Re-run 3-body solutions for top candidates.
    analyses_best : list of dict
        Matching analysis dicts (planet-frame).
    cfg : FullConfig
        Configuration (used for masses, radii, system name).
    num_b, num_angles : int
        2-body scan resolution  (total = num_b × num_angles per body).
    num_points : int
        Position samples per trajectory.
    padding_frac : float
        Fractional padding around 3-body extent.
    save_dir : str or None
        Directory to save to.
    dpi : int
        Output resolution.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per scattering body (star, planet).
    """
    from .twobody import TwoBodyEncounter, TrajectoryResult

    envelope = narrowed.get("envelope")
    if envelope is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No envelope — skipped",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        return [fig]

    # ------------------------------------------------------------------
    # 1.  Collect 3-body candidate positions (km) for axis limits
    # ------------------------------------------------------------------
    all_x, all_y = [], []
    for sol in sols_best:
        if sol is not None:
            x3 = sol.y[0] - sol.y[4]   # sat_x – planet_x  (planet frame)
            y3 = sol.y[1] - sol.y[5]   # sat_y – planet_y
            all_x.append(x3)
            all_y.append(y3)

    if all_x:
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
    else:
        # Fallback: use envelope b range
        span = envelope.b_max * 2
        all_x = np.array([-span, span])
        all_y = np.array([-span, span])

    x_lo, x_hi = float(all_x.min()), float(all_x.max())
    y_lo, y_hi = float(all_y.min()), float(all_y.max())
    # Ensure square with padding
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    half = max(x_hi - x_lo, y_hi - y_lo) * 0.5 * (1 + padding_frac)
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    # Convert limits to display units (×10⁹ km)
    SCALE = 1e9
    xlim_d = (xlim[0] / SCALE, xlim[1] / SCALE)
    ylim_d = (ylim[0] / SCALE, ylim[1] / SCALE)

    # ------------------------------------------------------------------
    # 2.  Build encounters & run scans
    # ------------------------------------------------------------------
    M_star_kg = cfg.system.M_star_Msun * M_SUN
    M_planet_kg = cfg.system.M_planet_Mjup * M_JUP
    R_star_km = getattr(cfg.system, "R_star_Rsun", 1.0) * R_SUN
    R_planet_km = getattr(cfg.system, "R_planet_Rjup", 1.155) * R_JUP

    bodies = [
        ("Star",   M_star_kg,  R_star_km),
        ("Planet", M_planet_kg, R_planet_km),
    ]

    # Envelope-derived scan parameters
    v_approach = 0.5 * (envelope.v_approach_min + envelope.v_approach_max)
    vstar0 = envelope.vstar0

    if envelope.b_min > 0:
        b_values = np.logspace(
            np.log10(envelope.b_min),
            np.log10(envelope.b_max),
            num_b,
        )
    else:
        b_values = np.linspace(envelope.b_min, envelope.b_max, num_b)

    angle_values = np.linspace(envelope.angle_min, envelope.angle_max, num_angles)
    r_start = 1.0e11  # km — far enough for asymptotic approach
    if hasattr(cfg, "two_body") and cfg.two_body is not None:
        r_start = cfg.two_body.r_start_km

    scan_results: Dict[str, List[TrajectoryResult]] = {}
    for label, M_kg, R_km in bodies:
        enc = TwoBodyEncounter(M_kg, G_KM, label=label.lower(), R_body_km=R_km)
        trajs, _energies, _grid = enc.scan_parameter_space(
            v_approach=v_approach,
            vstar0=vstar0,
            r_start=r_start,
            b_values=b_values,
            angle_values=angle_values,
            num_points=num_points,
        )
        scan_results[label] = trajs

    # ------------------------------------------------------------------
    # 3.  One figure per body
    # ------------------------------------------------------------------
    cmap = plt.cm.get_cmap("twilight")
    sys_name = getattr(cfg.system, "name", "")
    envelope_line = (
        f"Envelope: v ∈ [{envelope.v_approach_min:.1f}, {envelope.v_approach_max:.1f}] km/s   "
        f"b ∈ [{envelope.b_min:.2e}, {envelope.b_max:.2e}] km   "
        f"α ∈ [{np.degrees(envelope.angle_min):.0f}°, {np.degrees(envelope.angle_max):.0f}°]"
    )

    figs: List[plt.Figure] = []

    for label, _M, _R in bodies:
        trajs = scan_results[label]
        fig, ax = plt.subplots(figsize=(13, 11))

        # Colour normalisation from this body's energy range
        valid_e = np.array([t.orbital_energy for t in trajs if t.valid])
        if len(valid_e) == 0:
            ax.text(0.5, 0.5, f"{label}: no valid trajectories",
                    ha="center", va="center", fontsize=14,
                    transform=ax.transAxes)
            figs.append(fig)
            continue

        norm = plt.Normalize(vmin=valid_e.min(), vmax=valid_e.max())

        # 2-body trajectory curves
        for traj in trajs:
            if not traj.valid or len(traj.x_star) == 0:
                continue
            x = traj.x_star / SCALE
            y = traj.y_star / SCALE
            colour = cmap(norm(traj.orbital_energy))
            ax.plot(x, y, lw=0.8, color=colour, alpha=0.55, zorder=2)

        # 3-body candidate overlays
        for sol in sols_best:
            if sol is None:
                continue
            x3 = (sol.y[0] - sol.y[4]) / SCALE
            y3 = (sol.y[1] - sol.y[5]) / SCALE
            ax.plot(x3, y3, lw=1.6, color="cyan", alpha=0.85, zorder=5)

        # Scattering body marker at origin
        ax.plot(0, 0, "*", color="gold", markersize=20,
                markeredgecolor="black", markeredgewidth=1.0, zorder=10)

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.02)
        cb.set_label("½ΔV²  (km²/s² ≡ MJ/kg)", fontsize=12)

        # Legend
        legend_elements = [
            Line2D([0], [0], color="cyan", lw=2, label="3-body candidates"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markersize=14, markeredgecolor="black",
                   label=f"Scattering body ({label.lower()})"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

        # Axis cosmetics
        ax.set_xlim(xlim_d)
        ax.set_ylim(ylim_d)
        ax.set_xlabel("X  (×10⁹ km)", fontsize=12)
        ax.set_ylabel("Y  (×10⁹ km)", fontsize=12)
        ax.set_title(
            f"{label} Scattering Trajectory Tracks — {sys_name}\n"
            f"{len(trajs)} trajectories  |  {envelope_line}",
            fontsize=13, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")
        fig.tight_layout()

        if save_dir:
            tag = label.lower()
            p = Path(save_dir)
            fig.savefig(p / f"trajectory_tracks_{tag}.png",
                        dpi=dpi, bbox_inches="tight")

        figs.append(fig)

    return figs
