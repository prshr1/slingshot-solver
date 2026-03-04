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

from ..constants import G_KM, M_SUN, M_JUP, R_SUN, R_JUP

# ---------------------------------------------------------------------------
#  Import ground-truth physics engine from core
# ---------------------------------------------------------------------------
from ..core.twobody_scatter import (
    gravity_assist_no_burn as _ga_no_burn,
    gravity_assist_oberth as _ga_oberth,
    deltaV_lab as _deltaV_lab,
    _star_velocity_components,
)


def _subplot_grid_max_two(
    n_panels: int,
    panel_width: float = 6.2,
    panel_height: float = 5.6,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a subplot grid with at most 2 columns and hide unused axes."""
    n_panels = max(1, int(n_panels))
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes_flat


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
    vstar0: Any,
    r_far: float,
) -> Dict[str, np.ndarray]:
    """Compute b × α_inf Poincaré encounter grid.

    All units: km-kg-s.
    Returns dict with 2-D arrays: deltaV, theta, vinf, rp, b_grid, alpha_grid.
    """
    b_arr = np.logspace(np.log10(b_min), np.log10(b_max), num_b)
    alpha_arr = np.linspace(alpha_min_rad, alpha_max_rad, num_angle)
    B, A = np.meshgrid(b_arr, alpha_arr, indexing="ij")

    deltaV = np.full_like(B, np.nan)
    theta = np.full_like(B, np.nan)
    vinf_out = np.full_like(B, np.nan)
    rp = np.full_like(B, np.nan)
    vstar_x, vstar_y = _star_velocity_components(vstar0)
    vstar_vec = (vstar_x, vstar_y)

    for i in range(num_b):
        for j in range(num_angle):
            b_val = b_arr[i]
            alpha = alpha_arr[j]
            # Position at r_far along approach direction, offset by b perpendicular
            xm0 = r_far * np.cos(alpha) - b_val * np.sin(alpha)
            ym0 = r_far * np.sin(alpha) + b_val * np.cos(alpha)
            um0 = -v_inf * np.cos(alpha) + vstar_x
            vm0 = -v_inf * np.sin(alpha) + vstar_y
            try:
                res = _ga_no_burn(xm0, ym0, um0, vm0, vstar_vec, mu)
                deltaV[i, j] = _deltaV_lab(um0, vm0, res.umF, res.vmF)
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
    vstar0: Any,
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
    vstar_x, vstar_y = _star_velocity_components(vstar0)
    vstar_vec = (vstar_x, vstar_y)

    angle_rad = np.radians(approach_angle_deg)
    ux_rel = -v_approach * np.cos(angle_rad)
    uy_rel = -v_approach * np.sin(angle_rad)

    for i in range(num_x):
        for j in range(num_y):
            xm0 = r_encounter * np.cos(angle_rad) + x_arr[i]
            ym0 = r_encounter * np.sin(angle_rad) + y_arr[j]
            um0 = ux_rel + vstar_x
            vm0 = uy_rel + vstar_y
            try:
                res = _ga_no_burn(xm0, ym0, um0, vm0, vstar_vec, mu)
                deltaV[i, j] = _deltaV_lab(um0, vm0, res.umF, res.vmF)
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
    vstar0: Any,
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
    vstar_x, vstar_y = _star_velocity_components(vstar0)
    vstar_vec = (vstar_x, vstar_y)

    angle_rad = np.radians(approach_angle_deg)

    for i in range(num_b):
        for j in range(num_theta):
            bval = b_arr[i]
            tb = theta_arr[j]
            offset_x = bval * np.cos(tb)
            offset_y = bval * np.sin(tb)
            xm0 = r_start * np.cos(angle_rad) + offset_x
            ym0 = r_start * np.sin(angle_rad) + offset_y
            ux = -v_approach * np.cos(angle_rad) + vstar_x
            uy = -v_approach * np.sin(angle_rad) + vstar_y
            try:
                res = _ga_no_burn(xm0, ym0, ux, uy, vstar_vec, mu)
                deltaV[i, j] = _deltaV_lab(ux, uy, res.umF, res.vmF)
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
    vstar0_kms: Any = 10.0,
    num_b: int = 60,
    num_angle: int = 60,
    b_min_km: float = 1e7,
    b_max_km: float = 2e9,
    r_far_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate standalone Poincare (b vs alpha_inf) heatmaps."""
    mu = G_KM * M_body_kg
    grid = _compute_encounter_grid_poincare(
        mu=mu, num_b=num_b, num_angle=num_angle,
        b_min=b_min_km, b_max=b_max_km,
        alpha_min_rad=0.2, alpha_max_rad=np.pi - 0.2,
        v_inf=v_inf_kms, vstar0=vstar0_kms, r_far=r_far_km,
    )
    b_plot = grid["B"]  # keep in km for log-scale y-axis
    a_plot = np.degrees(grid["A"])

    figs: List[plt.Figure] = []
    tag = body_label.lower().replace(" ", "_")

    panels = [
        (grid["deltaV"], "Delta-V (km/s)", "hot_r", f"poincare_heatmap_delta_v_{tag}.png"),
        (grid["theta"], "Deflection theta (deg)", "viridis", f"poincare_heatmap_deflection_{tag}.png"),
        (grid["vinf"], "v_inf out (km/s)", "cool", f"poincare_heatmap_vinf_{tag}.png"),
        (grid["rp"], "Periapsis rp (km)", "plasma", f"poincare_heatmap_periapsis_{tag}.png"),
    ]

    for data, label, cmap, fname in panels:
        fig, ax = plt.subplots(figsize=(14, 8))
        cf = ax.pcolormesh(a_plot, b_plot, data, cmap=cmap, shading="gouraud")
        fig.colorbar(cf, ax=ax, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Approach angle alpha (deg)")
        ax.set_ylabel("Impact parameter b (km)")
        ax.set_title(f"Poincare map - {label} - {body_label}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        figs.append(fig)
        if save_dir:
            fig.savefig(Path(save_dir) / fname, dpi=dpi)

    fig2, ax = plt.subplots(figsize=(14, 8))
    cf = ax.pcolormesh(a_plot, b_plot, grid["deltaV"], cmap="hot_r", shading="gouraud")
    fig2.colorbar(cf, ax=ax, label="Delta-V (km/s)")
    ax.contour(a_plot, b_plot, grid["theta"], levels=10, colors="cyan", linewidths=1.5, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Approach angle alpha (deg)", fontsize=12)
    ax.set_ylabel("Impact parameter b (km)", fontsize=12)
    ax.set_title(f"Delta-V with deflection contours - {body_label}", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    figs.append(fig2)
    if save_dir:
        fig2.savefig(Path(save_dir) / f"poincare_heatmap_combined_{tag}.png", dpi=dpi)

    return figs


# ===================================================================
# 1b. 3-body Poincaré scatter (from MC integration results)
# ===================================================================

def plot_3body_poincare_scatter(
    mc: Dict[str, Any],
    save_dir: Optional[Path] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (14, 8),
) -> Dict[str, plt.Figure]:
    """Scatter plots of 3-body MC outcomes in initial-condition space.

    Unlike the analytic 2-body Poincaré maps, these show the *actual*
    integrated 3-body results, where the planet's presence breaks the
    symmetry and creates anisotropic structure.

    Axes: impact parameter b (AU) vs approach angle (deg), coloured by
    outcome metric (Δv, deflection, energy, star proximity).

    Parameters
    ----------
    mc : dict
        Monte-Carlo results dict from ``run_monte_carlo``.  Must contain
        ``sampling_params`` with ``impact_param_AU`` and ``angle_in_deg``.

    Returns
    -------
    dict  :  filename → Figure
    """
    sp = mc.get("sampling_params")
    if sp is None:
        return {}
    b_au = np.asarray(sp.get("impact_param_AU", []))
    angle_deg = np.asarray(sp.get("angle_in_deg", []))
    if len(b_au) == 0 or len(angle_deg) == 0:
        return {}

    ok = np.asarray(mc["ok"], dtype=bool)
    delta_v = np.asarray(mc["delta_v"])
    deflection = np.asarray(mc["deflection"])
    e_from_planet = np.asarray(mc.get("energy_from_planet_orbit",
                                       np.full(len(ok), np.nan)))
    r_star_min = np.asarray(mc.get("r_star_min",
                                    np.full(len(ok), np.nan)))
    R_star = mc.get("R_star", None)
    R_p = mc.get("R_p", None)

    # --- optional: v_mag for 3-axis colour (secondary) ---
    v_mag = np.asarray(sp.get("v_mag_kms", []))
    has_vmag = len(v_mag) == len(ok)

    # We only plot successful integrations
    mask = ok & np.isfinite(delta_v)
    b = b_au[mask]
    ang = angle_deg[mask]
    dv = delta_v[mask]
    defl = deflection[mask]
    ep = e_from_planet[mask]
    rstar = r_star_min[mask]
    vm = v_mag[mask] if has_vmag else None

    if len(b) < 3:
        return {}

    figs: Dict[str, plt.Figure] = {}

    # ----- helper -----
    def _make_scatter(colour_data, clabel, cmap, fname, *, vmin=None, vmax=None,
                      log_y=False, add_hexbin=False):
        fig, ax = plt.subplots(figsize=figsize)
        finite = np.isfinite(colour_data)
        b_f, ang_f, c_f = b[finite], ang[finite], colour_data[finite]
        if len(c_f) < 3:
            plt.close(fig)
            return
        if vmin is None:
            vmin = np.nanpercentile(c_f, 2)
        if vmax is None:
            vmax = np.nanpercentile(c_f, 98)
        if add_hexbin and len(c_f) > 200:
            hb = ax.hexbin(
                ang_f, b_f, C=c_f, gridsize=40, cmap=cmap,
                reduce_C_function=np.nanmedian, mincnt=1,
                linewidths=0.2, edgecolors="gray",
            )
            fig.colorbar(hb, ax=ax, label=clabel, shrink=0.85)
        else:
            sc = ax.scatter(
                ang_f, b_f, c=c_f, s=18, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=0.75,
                edgecolors="k", linewidths=0.15, zorder=3,
            )
            fig.colorbar(sc, ax=ax, label=clabel, shrink=0.85)
        if log_y and np.all(b_f > 0):
            ax.set_yscale("log")
        ax.set_xlabel("Approach angle (deg)", fontsize=12)
        ax.set_ylabel("Impact parameter b (AU)", fontsize=12)
        ax.set_title(
            f"3-body Poincaré — {clabel}  ({len(c_f)} particles)",
            fontsize=13, fontweight="bold",
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        figs[fname] = fig
        if save_dir:
            fig.savefig(Path(save_dir) / fname, dpi=dpi, bbox_inches="tight")

    # 1. Δv scatter
    _make_scatter(dv, "Δv (km/s)", "hot_r",
                  "poincare_3body_delta_v.png")
    # 2. |ΔV_vec| scatter (vector magnitude)
    dv_vec = np.asarray(mc.get("delta_v_vec", np.full(len(ok), np.nan)))
    dv_vec_ok = dv_vec[mask]
    if np.any(np.isfinite(dv_vec_ok)):
        _make_scatter(dv_vec_ok, "|ΔV_vec| (km/s)", "hot_r",
                      "poincare_3body_delta_v_vec.png")
    # 3. Deflection
    _make_scatter(np.abs(defl), "|Deflection| (deg)", "viridis",
                  "poincare_3body_deflection.png")
    # 4. Energy from planet orbit
    _make_scatter(ep, "ΔE planet orbit (km²/s²)", "coolwarm",
                  "poincare_3body_energy_planet.png")
    # 5. Star closest approach
    if R_star and R_star > 0:
        _make_scatter(rstar / R_star, "r_star_min / R★", "YlOrRd_r",
                      "poincare_3body_star_proximity.png")
    else:
        _make_scatter(rstar, "r_star_min (km)", "YlOrRd_r",
                      "poincare_3body_star_proximity.png")

    # 6. If v_mag available: 3-panel (angle vs b colored by Δv) sliced by v_mag
    if vm is not None and len(np.unique(vm)) > 3:
        v_terciles = np.percentile(vm, [33, 67])
        slices = [
            ("low v", vm <= v_terciles[0]),
            ("mid v", (vm > v_terciles[0]) & (vm <= v_terciles[1])),
            ("high v", vm > v_terciles[1]),
        ]
        fig_sl, axes_sl = plt.subplots(1, 3, figsize=(figsize[0] * 1.3, figsize[1]),
                                       sharey=True, sharex=True)
        vmin_dv = np.nanpercentile(dv, 2)
        vmax_dv = np.nanpercentile(dv, 98)
        for ax_i, (label, sl_mask) in zip(axes_sl, slices):
            b_s, ang_s, dv_s = b[sl_mask], ang[sl_mask], dv[sl_mask]
            sc = ax_i.scatter(
                ang_s, b_s, c=dv_s, s=20, cmap="hot_r",
                vmin=vmin_dv, vmax=vmax_dv, alpha=0.75,
                edgecolors="k", linewidths=0.15,
            )
            ax_i.set_title(f"{label} ({len(dv_s)} pts)", fontsize=11)
            ax_i.set_xlabel("Approach angle (deg)")
            ax_i.grid(True, alpha=0.2)
        axes_sl[0].set_ylabel("Impact parameter b (AU)")
        fig_sl.colorbar(sc, ax=axes_sl.tolist(), label="Δv (km/s)", shrink=0.8)
        fig_sl.suptitle(
            "3-body Poincaré sliced by approach speed",
            fontsize=13, fontweight="bold",
        )
        fig_sl.tight_layout(rect=[0, 0, 0.92, 0.95])
        fname_sl = "poincare_3body_vmag_slices.png"
        figs[fname_sl] = fig_sl
        if save_dir:
            fig_sl.savefig(Path(save_dir) / fname_sl, dpi=dpi, bbox_inches="tight")

    return figs


def plot_scattering_maps(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: Any = 10.0,
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
    """Generate standalone polar scattering maps for multiple approach angles."""
    mu = G_KM * M_body_kg
    figs: List[plt.Figure] = []
    tag = body_label.lower().replace(" ", "_")

    for angle_deg in approach_angles_deg:
        grid = _compute_encounter_grid_polar(
            mu=mu, num_b=num_b, num_theta=num_theta,
            b_min=b_min_km, b_max=b_max_km,
            v_approach=v_approach_kms, vstar0=vstar0_kms,
            approach_angle_deg=angle_deg, r_start=r_start_km,
        )
        b_plot = grid["Bmag"] / 1e6
        t_plot = np.degrees(grid["Theta_b"])
        suffix = f"{angle_deg:.0f}deg"

        panels = [
            (grid["deltaV"], "Delta-V (km/s)", "hot_r", "delta_v"),
            (grid["theta_defl"], "Deflection theta (deg)", "viridis", "deflection"),
            (grid["rp"], "Periapsis rp (km)", "plasma", "periapsis"),
        ]
        for data, label, cmap, slug in panels:
            fig, ax = plt.subplots(figsize=(14, 8))
            cf = ax.contourf(t_plot, b_plot, data, levels=30, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=label)
            ax.set_xlabel("Impact direction theta_b (deg)")
            ax.set_ylabel("Impact parameter b (10^6 km)")
            ax.set_title(f"Scattering map - {label} - {body_label} - approach {suffix}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            figs.append(fig)
            if save_dir:
                fig.savefig(Path(save_dir) / f"scattering_map_{tag}_{suffix}_{slug}.png", dpi=dpi)

    return figs


def plot_encounter_2d_cartesian(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: Any = 10.0,
    approach_angles_deg: List[float] = (0.0, 45.0, 85.0),
    num_xy: int = 70,
    xy_range_km: float = 3e9,
    r_encounter_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate standalone Cartesian (x, y) encounter heatmaps."""
    mu = G_KM * M_body_kg
    figs: List[plt.Figure] = []
    tag = body_label.lower().replace(" ", "_")

    for angle_deg in approach_angles_deg:
        grid = _compute_encounter_grid_cartesian(
            mu=mu, num_x=num_xy, num_y=num_xy,
            xy_range=xy_range_km, v_approach=v_approach_kms,
            vstar0=vstar0_kms, approach_angle_deg=angle_deg,
            r_encounter=r_encounter_km,
        )
        x_plot = grid["X"] / 1e6
        y_plot = grid["Y"] / 1e6
        suffix = f"{angle_deg:.0f}deg"

        panels = [
            (grid["deltaV"], "Delta-V (km/s)", "hot_r", "delta_v"),
            (grid["theta"], "Deflection theta (deg)", "viridis", "deflection"),
            (grid["rp"], "Periapsis rp (km)", "plasma", "periapsis"),
            (grid["b"], "Impact parameter b (km)", "cool", "impact_parameter"),
        ]
        for data, label, cmap, slug in panels:
            fig, ax = plt.subplots(figsize=(14, 8))
            cf = ax.contourf(x_plot, y_plot, data, levels=30, cmap=cmap)
            fig.colorbar(cf, ax=ax, label=label)
            ax.add_patch(Circle((0, 0), 0.1, fill=False, ec="cyan", lw=1.5))
            ax.set_aspect("equal")
            ax.set_xlabel("x offset (10^6 km)")
            ax.set_ylabel("y offset (10^6 km)")
            ax.set_title(f"2D encounter map - {label} - {body_label} - approach {suffix}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            figs.append(fig)
            if save_dir:
                fig.savefig(Path(save_dir) / f"encounter_2d_{tag}_{suffix}_{slug}.png", dpi=dpi)

    return figs


def plot_encounter_2d_trajectories(
    M_body_kg: float,
    v_approach_kms: float = 50.0,
    vstar0_kms: Any = 10.0,
    approach_angles_deg: List[float] = (0.0, 45.0, 85.0),
    num_xy: int = 60,
    xy_range_km: float = 3e9,
    r_encounter_km: float = 1e10,
    body_label: str = "Star",
    save_dir: Optional[Path] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Generate standalone delta-V heatmaps per approach angle."""
    mu = G_KM * M_body_kg
    figs: List[plt.Figure] = []
    tag = body_label.lower().replace(" ", "_")

    for angle_deg in approach_angles_deg:
        grid = _compute_encounter_grid_cartesian(
            mu=mu, num_x=num_xy, num_y=num_xy,
            xy_range=xy_range_km, v_approach=v_approach_kms,
            vstar0=vstar0_kms, approach_angle_deg=angle_deg,
            r_encounter=r_encounter_km,
        )
        x_plot = grid["X"] / 1e6
        y_plot = grid["Y"] / 1e6
        suffix = f"{angle_deg:.0f}deg"

        fig, ax = plt.subplots(figsize=(14, 8))
        cf = ax.contourf(x_plot, y_plot, grid["deltaV"], levels=30, cmap="hot_r")
        fig.colorbar(cf, ax=ax, label="Delta-V (km/s)")
        ax.add_patch(Circle((0, 0), 0.15, fill=True, fc="gold", ec="orange", lw=1.5, alpha=0.7))
        ax.set_aspect("equal")
        ax.set_title(f"2D encounter Delta-V - {body_label} - approach {suffix}", fontsize=13, fontweight="bold")
        ax.set_xlabel("x (10^6 km)")
        ax.set_ylabel("y (10^6 km)")
        fig.tight_layout()
        figs.append(fig)

        if save_dir:
            fig.savefig(Path(save_dir) / f"encounter_2d_{tag}_{suffix}_delta_v.png", dpi=dpi)

    return figs


def plot_oberth_comparison(
    M_body_kg: float,
    v_inf_kms: float = 50.0,
    vstar0_kms: Any = 10.0,
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
    """Compare no-burn vs Oberth burn with standalone figures."""
    mu = G_KM * M_body_kg
    b_arr = np.logspace(np.log10(b_min_km), np.log10(b_max_km), num_b)
    alpha_arr = np.linspace(0.1, np.pi - 0.1, num_angle)
    B, A = np.meshgrid(b_arr, alpha_arr, indexing="ij")

    dv_no = np.full_like(B, np.nan)
    dv_ob = np.full_like(B, np.nan)
    gain = np.full_like(B, np.nan)
    vstar_x, vstar_y = _star_velocity_components(vstar0_kms)
    vstar_vec = (vstar_x, vstar_y)

    for i in range(num_b):
        for j in range(num_angle):
            bval = b_arr[i]
            alpha = alpha_arr[j]
            # Position at r_far along approach direction, offset by b perpendicular
            xm0 = r_far_km * np.cos(alpha) - bval * np.sin(alpha)
            ym0 = r_far_km * np.sin(alpha) + bval * np.cos(alpha)
            um0 = -v_inf_kms * np.cos(alpha) + vstar_x
            vm0 = -v_inf_kms * np.sin(alpha) + vstar_y
            try:
                res_nb = _ga_no_burn(xm0, ym0, um0, vm0, vstar_vec, mu)
                dv_no[i, j] = _deltaV_lab(um0, vm0, res_nb.umF, res_nb.vmF)
                res_ob = _ga_oberth(xm0, ym0, um0, vm0, vstar_vec, mu, dv_burn_kms)
                dv_ob[i, j] = _deltaV_lab(um0, vm0, res_ob.umF, res_ob.vmF)
                gain[i, j] = dv_ob[i, j] - dv_no[i, j]
            except Exception:
                pass

    b_plot = B
    a_plot = np.degrees(A)
    figs: List[plt.Figure] = []
    tag = body_label.lower().replace(" ", "_")

    panels = [
        (dv_no, "Delta-V no-burn (km/s)", "hot_r", f"oberth_comparison_no_burn_{tag}.png"),
        (dv_ob, f"Delta-V Oberth burn Delta-v={dv_burn_kms} km/s", "hot_r", f"oberth_comparison_with_burn_{tag}.png"),
        (gain, "Oberth gain (km/s)", "RdYlGn", f"oberth_comparison_gain_{tag}.png"),
    ]
    for data, label, cmap, fname in panels:
        fig, ax = plt.subplots(figsize=(14, 8))
        cf = ax.pcolormesh(a_plot, b_plot, data, cmap=cmap, shading="gouraud")
        fig.colorbar(cf, ax=ax, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("alpha (deg)")
        ax.set_ylabel("Impact parameter b (km)")
        ax.set_title(f"Oberth comparison - {label} - {body_label}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        figs.append(fig)
        if save_dir:
            fig.savefig(Path(save_dir) / fname, dpi=dpi)

    fig2, ax = plt.subplots(figsize=(14, 8))
    cf = ax.pcolormesh(a_plot, b_plot, gain, cmap="RdYlGn", shading="gouraud")
    fig2.colorbar(cf, ax=ax, label="Oberth gain (km/s)")
    ax.contour(a_plot, b_plot, dv_no, levels=12, colors="gray", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("alpha (deg)", fontsize=12)
    ax.set_ylabel("Impact parameter b (km)", fontsize=12)
    ax.set_title(f"Oberth gain with no-burn contours - {body_label}", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    figs.append(fig2)
    if save_dir:
        fig2.savefig(Path(save_dir) / f"oberth_gain_{tag}.png", dpi=dpi)

    return figs


def plot_trajectory_tracks(
    narrowed: Dict[str, Any],
    sols_best: List[Any],
    analyses_best: List[Dict[str, Any]],
    cfg: Any,
    *,
    num_b: int = 140,
    num_angles: int = 120,
    num_points: int = 200,
    padding_frac: float = 0.20,
    max_overlay_tracks: int = 220,
    overlay_lines: bool = True,
    overlay_line_count: int = 90,
    gradient_mode: str = "hexbin",
    confidence_min_count: int = 2,
    fixed_energy_range: Optional[Tuple[float, float]] = None,
    hexbin_gridsize: int = 150,
    kde_sigma_bins: float = 2.0,
    time_frames: int = 48,
    export_phase_data: bool = True,
    export_time_data: bool = True,
    save_dir: Optional[str] = None,
    dpi: int = 150,
) -> List[plt.Figure]:
    """Trajectory + phase diagnostics for narrowed 2-body baselines.

    Parameters added for publication-grade gradients:
    - overlay_lines / overlay_line_count: optional trajectory overlays on top of gradients
    - gradient_mode: legacy | line_overlay | hexbin | kde | time_video
    - confidence_min_count: minimum per-bin support before a pixel is trusted
    - fixed_energy_range: optional (vmin, vmax) to enforce fixed normalization
    - hexbin_gridsize / kde_sigma_bins: estimator controls
    - time_frames/export_time_data: frame cube export for downstream animation
    """
    from ..core.twobody import TwoBodyEncounter, TrajectoryResult

    # Keep signature parity; currently unused directly.
    _ = analyses_best

    envelope = narrowed.get("envelope")
    if envelope is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No envelope - skipped",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        return [fig]

    mode = str(gradient_mode).strip().lower()
    valid_modes = {"legacy", "line_overlay", "hexbin", "kde", "time_video"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown gradient_mode '{gradient_mode}'. Valid: {sorted(valid_modes)}")

    min_count = max(0, int(confidence_min_count))
    count_thresh = max(1, min_count)
    overlay_line_count = max(0, int(overlay_line_count))
    hexbin_gridsize = max(20, int(hexbin_gridsize))
    kde_sigma_bins = max(0.1, float(kde_sigma_bins))
    time_frames = max(2, int(time_frames))

    M_star_kg = cfg.system.M_star_Msun * M_SUN
    M_planet_kg = cfg.system.M_planet_Mjup * M_JUP
    R_star_km = getattr(cfg.system, "R_star_Rsun", 1.0) * R_SUN
    R_planet_km = getattr(cfg.system, "R_planet_Rjup", 1.155) * R_JUP

    bodies = [
        ("star", "Star", M_star_kg, R_star_km, narrowed.get("star")),
        ("planet", "Planet", M_planet_kg, R_planet_km, narrowed.get("planet")),
    ]

    vstar_vec = envelope.vstar_vec if hasattr(envelope, "vstar_vec") else (0.0, envelope.vstar0)
    vstar_x, vstar_y = _star_velocity_components(vstar_vec)

    r_start = 1.0e11
    if hasattr(cfg, "two_body") and cfg.two_body is not None:
        r_start = cfg.two_body.r_start_km

    cmap = plt.cm.get_cmap("twilight" if mode == "legacy" else "viridis")
    sys_name = getattr(cfg.system, "name", "")
    envelope_line = (
        f"Envelope: v in [{envelope.v_approach_min:.1f}, {envelope.v_approach_max:.1f}] km/s, "
        f"b in [{envelope.b_min:.2e}, {envelope.b_max:.2e}] km, "
        f"alpha in [{np.degrees(envelope.angle_min):.0f} deg, {np.degrees(envelope.angle_max):.0f} deg]"
    )

    figs: List[plt.Figure] = []

    def _resolve_norm(values: np.ndarray) -> Tuple[plt.Normalize, float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            finite = np.array([0.0, 1.0], dtype=float)

        if fixed_energy_range is not None:
            vmin = float(fixed_energy_range[0])
            vmax = float(fixed_energy_range[1])
            if vmax <= vmin:
                raise ValueError(
                    f"fixed_energy_range must satisfy vmax > vmin, got {fixed_energy_range}"
                )
            return plt.Normalize(vmin=vmin, vmax=vmax), vmin, vmax

        if mode == "legacy":
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin = float(np.nanpercentile(finite, 2.0))
            vmax = float(np.nanpercentile(finite, 98.0))
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(finite))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return plt.Normalize(vmin=vmin, vmax=vmax), vmin, vmax

    def _extract_3body_tracks(tag: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        tracks: List[Tuple[np.ndarray, np.ndarray]] = []
        for sol in sols_best:
            if sol is None:
                continue
            y_raw = sol.get("y") if isinstance(sol, dict) else getattr(sol, "y", None)
            if y_raw is None:
                continue
            y = np.asarray(y_raw, dtype=float)
            if y.ndim != 2 or y.shape[0] < 10:
                continue
            if tag == "star":
                x_rel = y[8] - y[0]
                y_rel = y[9] - y[1]
            else:
                x_rel = y[8] - y[4]
                y_rel = y[9] - y[5]
            tracks.append((x_rel, y_rel))
        return tracks

    def _phase_grid_from_samples(
        sample_b: np.ndarray,
        sample_a: np.ndarray,
        sample_e: np.ndarray,
        sample_vx: np.ndarray,
        sample_vy: np.ndarray,
        b_edges: np.ndarray,
        a_edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nb = len(b_edges) - 1
        na = len(a_edges) - 1
        e_grid = np.full((nb, na), np.nan)
        vx_grid = np.full((nb, na), np.nan)
        vy_grid = np.full((nb, na), np.nan)
        n_grid = np.zeros((nb, na), dtype=int)

        bi = np.digitize(sample_b, b_edges) - 1
        ai = np.digitize(sample_a, a_edges) - 1
        valid = (
            (bi >= 0) & (bi < nb) &
            (ai >= 0) & (ai < na) &
            np.isfinite(sample_e)
        )
        idx = np.where(valid)[0]
        for k in idx:
            i, j = int(bi[k]), int(ai[k])
            n_grid[i, j] += 1
            if np.isnan(e_grid[i, j]) or sample_e[k] > e_grid[i, j]:
                e_grid[i, j] = sample_e[k]
                vx_grid[i, j] = sample_vx[k]
                vy_grid[i, j] = sample_vy[k]

        return e_grid, vx_grid, vy_grid, n_grid

    def _collect_trajectory_points(
        trajs: List[TrajectoryResult],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[TrajectoryResult], np.ndarray, np.ndarray, np.ndarray]:
        xs_list: List[np.ndarray] = []
        ys_list: List[np.ndarray] = []
        traj_kept: List[TrajectoryResult] = []
        energies: List[np.ndarray] = []
        for traj in trajs:
            if not traj.valid or len(traj.x_star) == 0:
                continue
            xs = np.asarray(traj.x_star, dtype=float)
            ys = np.asarray(traj.y_star, dtype=float)
            xs_list.append(xs)
            ys_list.append(ys)
            traj_kept.append(traj)
            energies.append(np.full(xs.shape, float(traj.orbital_energy), dtype=float))

        if not xs_list:
            return [], [], [], np.array([]), np.array([]), np.array([])

        xs_cat = np.concatenate(xs_list)
        ys_cat = np.concatenate(ys_list)
        es_cat = np.concatenate(energies)
        return xs_list, ys_list, traj_kept, xs_cat, ys_cat, es_cat

    def _grid_mean_energy(
        xs: np.ndarray,
        ys: np.ndarray,
        es: np.ndarray,
        r_vis_km: float,
        bins: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xy_range = [[-r_vis_km, r_vis_km], [-r_vis_km, r_vis_km]]
        sum_e, xedges, yedges = np.histogram2d(
            xs, ys, bins=bins, range=xy_range, weights=es,
        )
        cnt, _, _ = np.histogram2d(
            xs, ys, bins=bins, range=xy_range,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            e_grid = sum_e / cnt
        return e_grid, cnt, xedges, yedges

    def _build_time_evolution_cube(
        xs_list: List[np.ndarray],
        ys_list: List[np.ndarray],
        trajs_keep: List[TrajectoryResult],
        r_vis_km: float,
        bins: int,
        n_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xy_range = [[-r_vis_km, r_vis_km], [-r_vis_km, r_vis_km]]
        xedges = np.linspace(-r_vis_km, r_vis_km, bins + 1)
        yedges = np.linspace(-r_vis_km, r_vis_km, bins + 1)
        e_frames = np.full((n_frames, bins, bins), np.nan, dtype=float)
        c_frames = np.zeros((n_frames, bins, bins), dtype=float)

        for fi in range(n_frames):
            frac = fi / float(n_frames - 1)
            sum_e_f = np.zeros((bins, bins), dtype=float)
            cnt_f = np.zeros((bins, bins), dtype=float)
            for xs, ys, traj in zip(xs_list, ys_list, trajs_keep):
                n = len(xs)
                if n < 2:
                    continue
                idx = max(1, int(round(frac * (n - 1))))
                xseg = xs[: idx + 1]
                yseg = ys[: idx + 1]
                eseg = np.full(xseg.shape, float(traj.orbital_energy), dtype=float)
                hs, _, _ = np.histogram2d(xseg, yseg, bins=bins, range=xy_range, weights=eseg)
                hc, _, _ = np.histogram2d(xseg, yseg, bins=bins, range=xy_range)
                sum_e_f += hs
                cnt_f += hc

            with np.errstate(invalid="ignore", divide="ignore"):
                ef = sum_e_f / cnt_f
            ef[cnt_f < count_thresh] = np.nan
            e_frames[fi] = ef
            c_frames[fi] = cnt_f

        return e_frames, c_frames, xedges, yedges

    def _estimate_min_spacing(values: np.ndarray) -> float:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return 0.0
        uniq = np.unique(np.round(vals, 12))
        if uniq.size < 2:
            return 0.0
        diffs = np.diff(np.sort(uniq))
        diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
        if diffs.size == 0:
            return 0.0
        return float(np.min(diffs))

    def _expand_limits(lo: float, hi: float, frac: float, min_pad: float) -> Tuple[float, float]:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return lo, hi
        if hi < lo:
            lo, hi = hi, lo
        span = hi - lo
        pad = max(float(min_pad), float(span * frac))
        if span <= 0:
            pad = max(pad, 1.0)
        return lo - pad, hi + pad

    for tag, label, M_kg, R_km, narrowed_body in bodies:
        if narrowed_body is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5, f"{label}: no narrowed baseline result",
                ha="center", va="center", fontsize=14, transform=ax.transAxes,
            )
            figs.append(fig)
            continue

        sample_v = np.asarray(getattr(narrowed_body, "sample_v_approach", np.array([])), dtype=float)
        sample_b = np.asarray(getattr(narrowed_body, "sample_b", np.array([])), dtype=float)
        sample_a = np.asarray(getattr(narrowed_body, "sample_angle", np.array([])), dtype=float)
        sample_e = np.asarray(getattr(narrowed_body, "sample_energy", np.array([])), dtype=float)
        sample_vxf = np.asarray(getattr(narrowed_body, "sample_vx_final_rel", np.array([])), dtype=float)
        sample_vyf = np.asarray(getattr(narrowed_body, "sample_vy_final_rel", np.array([])), dtype=float)

        if sample_e.size == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5, f"{label}: no valid sample analytics",
                ha="center", va="center", fontsize=14, transform=ax.transAxes,
            )
            figs.append(fig)
            continue

        energy_norm, e_vmin, e_vmax = _resolve_norm(sample_e)

        # 1) phase map from narrowed sweep samples
        if envelope.b_min > 0 and (envelope.b_max / envelope.b_min) > 20:
            b_edges = np.logspace(np.log10(envelope.b_min), np.log10(envelope.b_max), num_b + 1)
        else:
            b_edges = np.linspace(envelope.b_min, envelope.b_max, num_b + 1)
        a_edges = np.linspace(envelope.angle_min, envelope.angle_max, num_angles + 1)

        e_grid, vx_grid, vy_grid, n_grid = _phase_grid_from_samples(
            sample_b, sample_a, sample_e, sample_vxf, sample_vyf, b_edges, a_edges
        )

        phase_mask = n_grid >= count_thresh
        e_grid_plot = np.where(phase_mask, e_grid, np.nan)
        vx_grid_plot = np.where(phase_mask, vx_grid, np.nan)
        vy_grid_plot = np.where(phase_mask, vy_grid, np.nan)

        b_cent = 0.5 * (b_edges[:-1] + b_edges[1:])
        a_cent = 0.5 * (a_edges[:-1] + a_edges[1:])
        A_deg_edges = np.degrees(a_edges)
        A_deg_cent = np.degrees(a_cent)

        fig_phase_e, ax_e = plt.subplots(figsize=(14, 8))

        pcm = ax_e.pcolormesh(
            A_deg_edges, b_edges / 1e6, e_grid_plot,
            shading="auto", cmap="viridis", norm=energy_norm,
        )
        cb1 = fig_phase_e.colorbar(pcm, ax=ax_e, pad=0.02)
        cb1.set_label("Scattering energy 0.5|DeltaV|^2 (km^2/s^2)")
        ax_e.set_xlabel("Approach angle alpha (deg)")
        ax_e.set_ylabel("Impact parameter b (10^6 km)")
        ax_e.set_title(f"{label}: scattering-energy gradient")
        ax_e.grid(True, alpha=0.2, linestyle="--")
        if np.any(phase_mask):
            ax_e.contour(
                A_deg_cent, b_cent / 1e6, phase_mask.astype(float),
                levels=[0.5], colors="white", linewidths=0.8, alpha=0.5,
            )
        fig_phase_e.suptitle(
            f"{label} phase diagnostics - {sys_name}\n"
            f"{envelope_line}\n"
            f"mode={mode}, conf>={count_thresh} samples/bin, energy=[{e_vmin:.2f}, {e_vmax:.2f}]",
            fontsize=12, fontweight="bold",
        )
        fig_phase_e.tight_layout()
        figs.append(fig_phase_e)

        fig_phase_v, ax_v = plt.subplots(figsize=(14, 8))
        heading = np.degrees(np.arctan2(vy_grid_plot, vx_grid_plot))
        pcm2 = ax_v.pcolormesh(
            A_deg_edges, b_edges / 1e6, heading,
            shading="auto", cmap="twilight", vmin=-180.0, vmax=180.0,
        )
        cb2 = fig_phase_v.colorbar(pcm2, ax=ax_v, pad=0.02)
        cb2.set_label("Final velocity heading (deg)")

        sb = max(1, len(b_cent) // 18)
        sa = max(1, len(a_cent) // 20)
        AQ, BQ = np.meshgrid(A_deg_cent[::sa], b_cent[::sb] / 1e6, indexing="xy")
        U = vx_grid_plot[::sb, ::sa]
        V = vy_grid_plot[::sb, ::sa]
        M = np.hypot(U, V)
        mask_q = np.isfinite(M) & (M > 0)
        Uq = np.zeros_like(U)
        Vq = np.zeros_like(V)
        Uq[mask_q] = U[mask_q] / M[mask_q]
        Vq[mask_q] = V[mask_q] / M[mask_q]
        ax_v.quiver(
            AQ, BQ, Uq, Vq,
            color="white", alpha=0.75, pivot="mid",
            scale=42, width=0.0022, headwidth=3.5,
        )

        ax_v.set_xlabel("Approach angle alpha (deg)")
        ax_v.set_ylabel("Impact parameter b (10^6 km)")
        ax_v.set_title(f"{label}: final velocity-direction field")
        ax_v.grid(True, alpha=0.2, linestyle="--")
        fig_phase_v.suptitle(
            f"{label} phase diagnostics - {sys_name}\n"
            f"{envelope_line}\n"
            f"mode={mode}, conf>={count_thresh} samples/bin, energy=[{e_vmin:.2f}, {e_vmax:.2f}]",
            fontsize=12, fontweight="bold",
        )
        fig_phase_v.tight_layout()
        figs.append(fig_phase_v)

        # 2) reconstruct representative trajectories for track-space rendering
        enc = TwoBodyEncounter(M_kg, G_KM, label=tag, R_body_km=R_km)
        n_valid = len(sample_e)
        order = np.argsort(sample_e)
        if n_valid > max_overlay_tracks:
            pick = order[np.linspace(0, n_valid - 1, max_overlay_tracks, dtype=int)]
        else:
            pick = order

        trajs: List[TrajectoryResult] = []
        for k in pick:
            v_approach = float(sample_v[k])
            angle = float(sample_a[k])
            b_mag = float(sample_b[k])
            vx = v_approach * np.cos(angle)
            vy = v_approach * np.sin(angle)
            perp = angle + np.pi / 2.0
            xm0 = -np.cos(angle) * r_start + b_mag * np.cos(perp)
            ym0 = -np.sin(angle) * r_start + b_mag * np.sin(perp)
            um0 = vx + vstar_x
            vm0 = vy + vstar_y
            tr = enc.compute_trajectory(xm0, ym0, um0, vm0, (vstar_x, vstar_y), num_points=num_points)
            if tr.valid and len(tr.x_star) > 0:
                trajs.append(tr)

        fig, ax = plt.subplots(figsize=(14, 8))
        valid_e = np.array([t.orbital_energy for t in trajs if t.valid], dtype=float)
        if valid_e.size == 0:
            ax.text(
                0.5, 0.5, f"{label}: no representative trajectories",
                ha="center", va="center", fontsize=14, transform=ax.transAxes,
            )
            figs.append(fig)
            continue

        x_pool: List[np.ndarray] = [t.x_star for t in trajs if t.valid]
        y_pool: List[np.ndarray] = [t.y_star for t in trajs if t.valid]
        tracks_3b = _extract_3body_tracks(tag)
        for x3, y3 in tracks_3b:
            x_pool.append(x3)
            y_pool.append(y3)

        if x_pool:
            xx = np.concatenate(x_pool)
            yy = np.concatenate(y_pool)
            rr = np.hypot(xx, yy)
            rr = rr[np.isfinite(rr)]
            if rr.size > 0:
                r_vis = float(np.percentile(rr, 96.0))
            else:
                r_vis = float(envelope.b_max * 1.5)
        else:
            r_vis = float(envelope.b_max * 1.5)

        r_vis = max(r_vis, float(R_km * 3.0), float(envelope.b_max * 0.8))
        r_vis *= (1.0 + padding_frac)
        SCALE = 1e6
        grad_pad_frac = max(0.05, min(0.30, 0.5 * float(padding_frac)))
        xlim_grad_km: Tuple[float, float] = (-r_vis, r_vis)
        ylim_grad_km: Tuple[float, float] = (-r_vis, r_vis)

        xs_list, ys_list, traj_keep, xs_cat, ys_cat, es_cat = _collect_trajectory_points(trajs)
        time_cube_energy = None
        time_cube_count = None
        time_xedges = None
        time_yedges = None

        if xs_cat.size > 0:
            if mode == "legacy":
                # Legacy style: clean line-rendered trajectories colored by specific orbital energy.
                for traj in trajs:
                    if not traj.valid or len(traj.x_star) == 0:
                        continue
                    x = traj.x_star / SCALE
                    y = traj.y_star / SCALE
                    colour = cmap(energy_norm(traj.orbital_energy))
                    ax.plot(x, y, lw=1.35, color=colour, alpha=0.72, zorder=2)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=energy_norm)
                sm.set_array([])
                cb_bg = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.02)
            elif mode == "hexbin":
                hb = ax.hexbin(
                    xs_cat / SCALE,
                    ys_cat / SCALE,
                    C=es_cat,
                    reduce_C_function=np.mean,
                    gridsize=hexbin_gridsize,
                    extent=(-r_vis / SCALE, r_vis / SCALE, -r_vis / SCALE, r_vis / SCALE),
                    mincnt=count_thresh,
                    cmap="viridis",
                    norm=energy_norm,
                    linewidths=0.0,
                    alpha=0.95,
                )
                cb_bg = fig.colorbar(hb, ax=ax, shrink=0.78, pad=0.02)
                offsets = np.asarray(hb.get_offsets(), dtype=float)
                if offsets.size > 0:
                    off_x = offsets[:, 0] * SCALE
                    off_y = offsets[:, 1] * SCALE
                    finite_off = np.isfinite(off_x) & np.isfinite(off_y)
                    if np.any(finite_off):
                        off_x = off_x[finite_off]
                        off_y = off_y[finite_off]
                        step_x = _estimate_min_spacing(off_x)
                        step_y = _estimate_min_spacing(off_y)
                        xlo, xhi = _expand_limits(
                            float(np.min(off_x)),
                            float(np.max(off_x)),
                            frac=grad_pad_frac,
                            min_pad=max(float(R_km * 0.25), step_x * 0.75, 1.0),
                        )
                        ylo, yhi = _expand_limits(
                            float(np.min(off_y)),
                            float(np.max(off_y)),
                            frac=grad_pad_frac,
                            min_pad=max(float(R_km * 0.25), step_y * 0.75, 1.0),
                        )
                        xlim_grad_km = (xlo, xhi)
                        ylim_grad_km = (ylo, yhi)
            else:
                bins = max(80, int(round(hexbin_gridsize * 1.25)))
                e_bg, cnt, xedges, yedges = _grid_mean_energy(xs_cat, ys_cat, es_cat, r_vis, bins)

                if mode in {"kde", "time_video"}:
                    sum_e, _, _ = np.histogram2d(
                        xs_cat,
                        ys_cat,
                        bins=bins,
                        range=[[-r_vis, r_vis], [-r_vis, r_vis]],
                        weights=es_cat,
                    )
                    sum_e_s = gaussian_filter(sum_e, sigma=kde_sigma_bins, mode="nearest")
                    cnt_s = gaussian_filter(cnt.astype(float), sigma=kde_sigma_bins, mode="nearest")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        e_bg = sum_e_s / cnt_s
                    e_bg[cnt_s < count_thresh] = np.nan
                else:
                    e_bg[cnt < count_thresh] = np.nan

                if mode == "time_video":
                    time_cube_energy, time_cube_count, time_xedges, time_yedges = _build_time_evolution_cube(
                        xs_list, ys_list, traj_keep, r_vis, bins, time_frames
                    )
                    e_bg = time_cube_energy[-1]

                pcm_bg = ax.pcolormesh(
                    xedges / SCALE,
                    yedges / SCALE,
                    e_bg.T,
                    shading="auto",
                    cmap="viridis",
                    norm=energy_norm,
                    alpha=0.92,
                )
                cb_bg = fig.colorbar(pcm_bg, ax=ax, shrink=0.78, pad=0.02)
                finite_bg = np.isfinite(e_bg)
                if np.any(finite_bg):
                    ix, iy = np.where(finite_bg)
                    xlo = float(xedges[int(np.min(ix))])
                    xhi = float(xedges[int(np.max(ix)) + 1])
                    ylo = float(yedges[int(np.min(iy))])
                    yhi = float(yedges[int(np.max(iy)) + 1])
                    xlim_grad_km = _expand_limits(
                        xlo,
                        xhi,
                        frac=grad_pad_frac,
                        min_pad=max(float(R_km * 0.25), 1.0),
                    )
                    ylim_grad_km = _expand_limits(
                        ylo,
                        yhi,
                        frac=grad_pad_frac,
                        min_pad=max(float(R_km * 0.25), 1.0),
                    )

            if mode == "legacy":
                cb_bg.set_label("Orbital Energy (km^2/s^2 ≡ MJ/kg)", fontsize=12)
            else:
                cb_bg.set_label("Scattering energy 0.5|DeltaV|^2 (km^2/s^2)", fontsize=12)

        xlo_plot, xhi_plot = xlim_grad_km
        ylo_plot, yhi_plot = ylim_grad_km
        body_pad = max(float(R_km * 1.2), 1.0)
        xlo_plot = min(xlo_plot, -body_pad)
        xhi_plot = max(xhi_plot, body_pad)
        ylo_plot = min(ylo_plot, -body_pad)
        yhi_plot = max(yhi_plot, body_pad)

        if mode != "legacy":
            gx = np.linspace(xlo_plot, xhi_plot, 260)
            gy = np.linspace(ylo_plot, yhi_plot, 260)
            GX, GY = np.meshgrid(gx, gy, indexing="xy")
            GR = np.hypot(GX, GY)
            GR = np.maximum(GR, max(R_km, 1.0))
            phi = (G_KM * M_kg) / GR
            lev_lo = np.percentile(phi, 20)
            lev_hi = np.percentile(phi, 99.5)
            if lev_hi > lev_lo:
                levels = np.geomspace(max(lev_lo, 1e-9), lev_hi, 10)
                ax.contour(
                    GX / SCALE,
                    GY / SCALE,
                    phi,
                    levels=levels,
                    colors="white",
                    alpha=0.16,
                    linewidths=0.8,
                    zorder=1,
                )

        if mode != "legacy" and overlay_lines and overlay_line_count > 0:
            overlay_trajs = trajs
            if len(trajs) > overlay_line_count:
                stride = max(1, len(trajs) // overlay_line_count)
                overlay_trajs = trajs[::stride]
            line_alpha = 0.35 if mode == "line_overlay" else 0.20
            for traj in overlay_trajs:
                x = traj.x_star / SCALE
                y = traj.y_star / SCALE
                colour = cmap(energy_norm(traj.orbital_energy))
                ax.plot(x, y, lw=0.65, color=colour, alpha=line_alpha, zorder=2)

        for x3, y3 in tracks_3b:
            ax.plot(x3 / SCALE, y3 / SCALE, lw=1.6, color="cyan", alpha=0.85, zorder=5)

        if mode == "legacy":
            ax.plot(
                0, 0, "*", color="black", markersize=20,
                markeredgecolor="white", markeredgewidth=1.5, zorder=10,
            )
        else:
            ax.plot(
                0, 0, "*", color="gold", markersize=20,
                markeredgecolor="black", markeredgewidth=1.0, zorder=10,
            )

        if mode == "legacy":
            legend_elements = [
                Line2D([0], [0], color="cyan", lw=2, label="3-body candidates"),
                Line2D(
                    [0], [0], marker="*", color="w", markerfacecolor="black",
                    markersize=14, markeredgecolor="white",
                    label=f"Scattering body ({label.lower()})",
                ),
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color="cyan", lw=2, label="3-body candidates"),
                Line2D(
                    [0], [0], marker="*", color="w", markerfacecolor="gold",
                    markersize=14, markeredgecolor="black",
                    label=f"Scattering body ({label.lower()})",
                ),
            ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

        ax.set_xlim((xlo_plot / SCALE, xhi_plot / SCALE))
        ax.set_ylim((ylo_plot / SCALE, yhi_plot / SCALE))
        ax.set_xlabel("X relative (10^6 km)", fontsize=12)
        ax.set_ylabel("Y relative (10^6 km)", fontsize=12)
        if mode == "legacy":
            ax.set_title(
                f"Trajectory Tracks Colored by Specific Orbital Energy | {label} Frame",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.set_title(
                f"{label} scattering tracks + energy gradient - {sys_name}\n"
                f"{len(trajs)} representative trajectories | mode={mode} | "
                f"overlay={'on' if (overlay_lines and overlay_line_count > 0) else 'off'}"
                f"{f'({overlay_line_count})' if (overlay_lines and overlay_line_count > 0) else ''} | "
                f"conf>={count_thresh} | "
                f"energy=[{e_vmin:.2f}, {e_vmax:.2f}]",
                fontsize=13,
                fontweight="bold",
            )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")
        fig.tight_layout()

        if save_dir:
            p = Path(save_dir)
            fig.savefig(p / f"trajectory_tracks_{tag}.png", dpi=dpi)
            fig_phase_e.savefig(p / f"trajectory_phase_energy_{tag}.png", dpi=dpi)
            fig_phase_v.savefig(p / f"trajectory_phase_heading_{tag}.png", dpi=dpi)
            if export_phase_data:
                np.savez(
                    p / f"trajectory_phase_data_{tag}.npz",
                    angle_edges_rad=a_edges,
                    b_edges_km=b_edges,
                    energy_grid=e_grid_plot,
                    heading_grid_deg=np.degrees(np.arctan2(vy_grid_plot, vx_grid_plot)),
                    vx_final_grid=vx_grid_plot,
                    vy_final_grid=vy_grid_plot,
                    count_grid=n_grid,
                    sample_v_approach=sample_v,
                    sample_b=sample_b,
                    sample_angle=sample_a,
                    sample_energy=sample_e,
                    sample_vx_final_rel=sample_vxf,
                    sample_vy_final_rel=sample_vyf,
                    vstar_vec=np.array([vstar_x, vstar_y], dtype=float),
                    r_start_km=float(r_start),
                    body=tag,
                    gradient_mode=mode,
                    confidence_min_count=int(count_thresh),
                    energy_vmin=float(e_vmin),
                    energy_vmax=float(e_vmax),
                )

            if (
                export_time_data
                and mode == "time_video"
                and time_cube_energy is not None
                and time_cube_count is not None
                and time_xedges is not None
                and time_yedges is not None
            ):
                np.savez(
                    p / f"trajectory_time_data_{tag}.npz",
                    energy_frames=time_cube_energy,
                    count_frames=time_cube_count,
                    x_edges_km=time_xedges,
                    y_edges_km=time_yedges,
                    frame_count=int(time_frames),
                    confidence_min_count=int(count_thresh),
                    energy_vmin=float(e_vmin),
                    energy_vmax=float(e_vmax),
                    body=tag,
                )

        figs.append(fig)

    return figs

