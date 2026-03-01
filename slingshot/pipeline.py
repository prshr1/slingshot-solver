"""
Unified pipeline orchestrator.

Usage:
    from slingshot.pipeline import run_pipeline
    results = run_pipeline("configs/config.yaml")

Each phase can also be called independently for debugging.
"""

import numpy as np
import pickle
import dataclasses
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt

from .config import load_config, save_config, FullConfig
from .console import configure_console_streams, safe_print as print
from .constants import G_KM, M_SUN, M_JUP, R_JUP, R_SUN, AU_KM
from .dynamics import simulate_3body, init_hot_jupiter_barycentric
from .analysis import analyze_trajectory, extract_encounter_states
from .monte_carlo import (
    run_monte_carlo,
    select_top_indices,
    select_pareto_indices,
    select_weighted_indices,
)
from .plotting import (
    plot_mc_summary_individual,
    plot_sampling_parameter_distributions,
    plot_best_candidate_with_bodies,
    plot_velocity_phase_space_individual,
    plot_star_proximity_distribution_individual,
    plot_planet_frame_diagnostics_individual,
    plot_multi_candidate_overlay,
    plot_rejection_breakdown,
    plot_parameter_correlations_individual,
    plot_energy_cdf,
    plot_publication_objectives_individual,
    plot_candidate_ranking_diagnostics_individual,
)
from .baselines import compare_3body_with_baselines
from .narrowed_baselines import compute_narrowed_baselines
from .comparison import compare_2body_3body, print_comparison
from .animation import generate_all_animations


# ===================================================================
# Helpers
# ===================================================================

def _strip_dataclasses(obj):
    """Deep-convert dataclass instances to plain dicts for pickling."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _strip_dataclasses(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _strip_dataclasses(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_dataclasses(v) for v in obj]
    return obj


def _derive_physics(cfg: FullConfig):
    """Derive SI physical quantities from config."""
    m_star = cfg.system.M_star_Msun * M_SUN
    m_p = cfg.system.M_planet_Mjup * M_JUP
    R_p = cfg.system.R_planet_Rjup * R_JUP
    R_star = cfg.system.R_star_Rsun * R_SUN
    return m_star, m_p, R_p, R_star


def _resolve_two_body_star_velocity(
    cfg: FullConfig,
    mc: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Resolve 2-body baseline star velocity vector from 3-body state.

    For consistency across comparisons, pipeline-generated 2-body diagnostics
    use the same initial star barycentric velocity as the 3-body run whenever MC
    state is available.
    """
    tb_cfg = cfg.two_body
    v_cfg = (0.0, tb_cfg.vstar0_kms if tb_cfg else 10.0)

    if mc is None or "Y_sp0" not in mc:
        return v_cfg

    Y_sp0 = mc["Y_sp0"]
    v_star_init = (float(Y_sp0[2]), float(Y_sp0[3]))

    if verbose:
        dvx = v_star_init[0] - v_cfg[0]
        dvy = v_star_init[1] - v_cfg[1]
        if np.hypot(dvx, dvy) > 1e-9:
            print(
                "  2-body vstar sync: "
                f"config=({v_cfg[0]:.3f}, {v_cfg[1]:.3f}) -> "
                f"3-body init=({v_star_init[0]:.3f}, {v_star_init[1]:.3f}) km/s"
            )

    return v_star_init


# ===================================================================
# Phase functions — each callable independently
# ===================================================================

def phase_monte_carlo(cfg: FullConfig, verbose: bool = True) -> Dict[str, Any]:
    """Phase 1: Monte Carlo sweep.

    Returns
    -------
    dict
        Monte Carlo results dict from ``run_monte_carlo``.
    """
    m_star, m_p, R_p, R_star = _derive_physics(cfg)

    if verbose:
        print(f"═══ Phase 1: Monte Carlo ({cfg.pipeline.N_particles} particles) ═══")

    mc = run_monte_carlo(
        N=cfg.pipeline.N_particles,
        t_span=(0.0, cfg.pipeline.t_mc_max_sec),
        m_star=m_star,
        m_p=m_p,
        R_p=R_p,
        frame="barycentric",
        sampling_mode=cfg.sampling.mode,
        n_parallel=cfg.pipeline.n_parallel,
        verbose=verbose,
        # Sampling bounds
        v_mag_min=cfg.sampling.v_mag_min_kms,
        v_mag_max=cfg.sampling.v_mag_max_kms,
        impact_param_min_AU=cfg.sampling.impact_param_min_AU,
        impact_param_max_AU=cfg.sampling.impact_param_max_AU,
        angle_in_min_deg=cfg.sampling.angle_in_min_deg,
        angle_in_max_deg=cfg.sampling.angle_in_max_deg,
        r_init_AU=cfg.sampling.r_init_AU,
        # Optional system bulk velocity (Galilean boost)
        bulk_velocity_vx_kms=cfg.system.bulk_velocity_vx_kms,
        bulk_velocity_vy_kms=cfg.system.bulk_velocity_vy_kms,
        # Numerical
        rtol=cfg.numerical.rtol,
        atol=cfg.numerical.atol,
        r_far_factor=cfg.numerical.r_far_factor,
        min_clearance_factor=cfg.numerical.min_clearance_factor,
        bary_unbound_requirement=cfg.sampling.bary_unbound_requirement,
        flyby_r_min_max_hill=cfg.numerical.flyby_r_min_max_hill,
        escape_radius_factor=cfg.numerical.escape_radius_factor,
        ode_method=cfg.numerical.ode_method,
        softening_km=cfg.numerical.softening_km,
        # Star filter
        star_min_clearance_Rstar=cfg.numerical.star_min_clearance_Rstar,
        R_star_Rsun=cfg.system.R_star_Rsun,
    )
    return mc


def phase_select(cfg: FullConfig, mc: Dict[str, Any],
                 verbose: bool = True) -> np.ndarray:
    """Phase 2: Select top candidates.

    Returns
    -------
    np.ndarray
        Indices of selected particles.
    """
    if verbose:
        print(f"\n═══ Phase 2: Select top {cfg.pipeline.top_frac*100:.0f}% ═══")
    mode = cfg.pipeline.select_mode

    if mode == "single":
        top_idx = select_top_indices(
            mc,
            top_frac=cfg.pipeline.top_frac,
            min_top=cfg.pipeline.min_top,
            metric=cfg.pipeline.select_metric,
            sign=cfg.pipeline.select_sign,
        )
        if verbose:
            print(
                "  Selection mode: single  "
                f"(metric={cfg.pipeline.select_metric}, sign={cfg.pipeline.select_sign})"
            )
    else:
        objectives = [
            {"metric": o.metric, "sign": o.sign, "weight": o.weight}
            for o in cfg.pipeline.selection_objectives
        ]

        if mode == "pareto":
            top_idx = select_pareto_indices(
                mc,
                objectives=objectives,
                top_frac=cfg.pipeline.top_frac,
                min_top=cfg.pipeline.min_top,
                tie_break_normalization=cfg.pipeline.weighted_normalization,
            )
        elif mode == "weighted":
            top_idx = select_weighted_indices(
                mc,
                objectives=objectives,
                top_frac=cfg.pipeline.top_frac,
                min_top=cfg.pipeline.min_top,
                normalization=cfg.pipeline.weighted_normalization,
            )
        else:
            raise ValueError(f"Unknown select_mode: {mode}")

        if verbose:
            obj_txt = ", ".join(
                f"{o['metric']}:{o['sign']}@{o['weight']:.2g}" for o in objectives
            )
            print(
                f"  Selection mode: {mode}  "
                f"(norm={cfg.pipeline.weighted_normalization}; objectives={obj_txt})"
            )

    if verbose:
        print(f"  Selected {len(top_idx)} candidates")
    return top_idx


def phase_rerun(
    cfg: FullConfig,
    mc: Dict[str, Any],
    top_idx: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase 3: High-resolution re-run of top candidates.

    Returns
    -------
    dict
        ``sols_best``, ``analyses_best``, ``Y0_best``, ``star_rejected``.
    """
    m_star, m_p, R_p, R_star = _derive_physics(cfg)

    if verbose:
        print(f"\n═══ Phase 3: Re-run {len(top_idx)} candidates ═══")

    star_clearance_Rstar = cfg.numerical.star_min_clearance_Rstar
    star_clearance_km = (star_clearance_Rstar * R_star
                         if star_clearance_Rstar is not None else None)

    Y_sp0 = mc["Y_sp0"]
    sat_states = mc["sat_states"]

    sols_best: List = []
    analyses_best: List = []
    Y0_best: List = []
    star_rejected = 0

    for i, idx in enumerate(top_idx):
        if verbose and i % max(1, len(top_idx) // 10) == 0:
            print(f"  Progress: {i + 1}/{len(top_idx)}")

        xs, ys, vxs, vys, xp, yp, vxp, vyp = Y_sp0
        x3, y3, vx3, vy3 = sat_states[idx]
        Y0 = np.array([xs, ys, vxs, vys, xp, yp, vxp, vyp,
                        x3, y3, vx3, vy3], dtype=float)

        r0_bary = np.hypot(x3, y3)
        escape_r = r0_bary * cfg.numerical.escape_radius_factor

        sol = simulate_3body(
            Y0,
            (0.0, cfg.pipeline.t_best_max_sec),
            m_star=m_star,
            m_p=m_p,
            n_eval=cfg.pipeline.n_eval_best,
            rtol=cfg.numerical.rtol,
            atol=cfg.numerical.atol,
            escape_radius_km=escape_r,
            method=cfg.numerical.ode_method,
            softening_km=cfg.numerical.softening_km,
        )

        ana = analyze_trajectory(
            sol, frame="barycentric",
            m_star=m_star, m_p=m_p, R_p=R_p,
            r_far_factor=cfg.numerical.r_far_factor,
            min_clearance_factor=cfg.numerical.min_clearance_factor,
        )

        # Post-hoc star filter
        if ana is not None and star_clearance_km is not None:
            enc = ana.get("encounter")
            if enc is not None and enc.r_star_min is not None:
                if enc.r_star_min < star_clearance_km:
                    ana = None
                    star_rejected += 1

        sols_best.append(sol)
        analyses_best.append(ana)
        Y0_best.append(Y0)

    valid_count = sum(1 for a in analyses_best if a is not None)
    if verbose:
        print(f"  Valid: {valid_count}/{len(top_idx)}")
        if star_rejected > 0:
            print(f"  Star-proximity rejected: {star_rejected}")

    return {
        "sols_best": sols_best,
        "analyses_best": analyses_best,
        "Y0_best": Y0_best,
        "star_rejected": star_rejected,
    }


def phase_best_selection(
    analyses_best: List[Optional[Dict]],
    top_idx: np.ndarray,
    sols_best: list,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase 4: Identify best candidates by scalar Δv and vector |ΔV_vec|.

    Returns
    -------
    dict
        ``best_ana``, ``best_sol``, ``best_idx``,
        ``best_vec_ana``, ``best_vec_sol``, ``best_vec_idx``.
    """
    if verbose:
        print(f"\n═══ Phase 4: Best selection ═══")

    valid = [(i, a) for i, a in enumerate(analyses_best) if a is not None]
    if not valid:
        print("  ERROR: No valid candidates!")
        return {"best_ana": None, "best_sol": None, "best_idx": None,
                "best_vec_ana": None, "best_vec_sol": None, "best_vec_idx": None}

    # Best by scalar Δv
    best_i = max(valid, key=lambda x: x[1]["delta_v"])[0]
    # Best by ½|ΔV_vec|²
    best_vec_i = max(valid, key=lambda x: x[1].get("energy_half_dv_vec_sq", 0))[0]

    result = {
        "best_ana": analyses_best[best_i],
        "best_sol": sols_best[best_i],
        "best_idx": int(top_idx[best_i]),
        "best_vec_ana": analyses_best[best_vec_i],
        "best_vec_sol": sols_best[best_vec_i],
        "best_vec_idx": int(top_idx[best_vec_i]),
    }

    if verbose:
        ba = result["best_ana"]
        print(f"  Best scalar Δv: MC#{result['best_idx']}  "
              f"Δv={ba['delta_v']:.2f} km/s  "
              f"|ΔV_vec|={ba.get('delta_v_vec', 0):.2f}")
        bva = result["best_vec_ana"]
        print(f"  Best |ΔV_vec|²: MC#{result['best_vec_idx']}  "
              f"|ΔV_vec|={bva.get('delta_v_vec', 0):.2f}  "
              f"½|ΔV|²={bva.get('energy_half_dv_vec_sq', 0):.2f}")

    return result


def phase_baselines(
    cfg: FullConfig,
    analyses_best: List[Optional[Dict]],
    best_vec_ana: Optional[Dict],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Phase 5: Narrowed 2-body baselines + comparison.

    Returns
    -------
    dict
        ``narrowed``, ``comparison``.
    """
    if verbose:
        print(f"\n═══ Phase 5: Narrowed 2-body baselines ═══")

    valid_analyses = [a for a in analyses_best if a is not None]
    if not valid_analyses or best_vec_ana is None:
        print("  Skipped — no valid candidates")
        return {"narrowed": None, "comparison": None}

    tb_cfg = cfg.two_body
    padding = tb_cfg.padding_factor if tb_cfg else 1.5
    num_v = tb_cfg.num_v if tb_cfg else 20
    num_b = tb_cfg.num_b_narrow if tb_cfg else 100
    num_ang = tb_cfg.num_angles_narrow if tb_cfg else 100

    narrowed = compute_narrowed_baselines(
        analyses_top=valid_analyses,
        cfg=cfg,
        padding_factor=padding,
        num_v=num_v,
        num_b=num_b,
        num_angles=num_ang,
        verbose=verbose,
    )

    # Extract 2-body metrics
    E_star = narrowed["star"].max_energy_half_dv_vec_sq if narrowed.get("star") else None
    E_planet = narrowed["planet"].max_energy_half_dv_vec_sq if narrowed.get("planet") else None
    dv_vec_star = narrowed["star"].max_deltaV_vec if narrowed.get("star") else None
    dv_vec_planet = narrowed["planet"].max_deltaV_vec if narrowed.get("planet") else None

    # 3-body best
    E_3body = best_vec_ana.get("energy_half_dv_vec_sq", 0.0)
    dv_vec_3body = best_vec_ana.get("delta_v_vec", 0.0)

    comp = compare_2body_3body(
        energy_2body_star=narrowed["star"].max_epsilon if narrowed.get("star") else None,
        energy_2body_planet=narrowed["planet"].max_epsilon if narrowed.get("planet") else None,
        energy_3body=E_3body,
        dv_2body_star=dv_vec_star,
        dv_2body_planet=dv_vec_planet,
        dv_3body=best_vec_ana.get("delta_v"),
        dv_vec_2body_star=dv_vec_star,
        dv_vec_2body_planet=dv_vec_planet,
        dv_vec_3body=dv_vec_3body,
        envelope_summary=narrowed["envelope"].summary() if narrowed.get("envelope") else None,
        baseline_mode="narrowed",
    )

    if verbose:
        print_comparison(comp)

    return {"narrowed": narrowed, "comparison": comp}


def phase_plots(
    cfg: FullConfig,
    mc: Dict[str, Any],
    top_idx: np.ndarray,
    rerun: Dict[str, Any],
    best: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
    baselines: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Phase 6: Generate all diagnostic plots.

    Returns list of saved file paths.
    """
    if verbose:
        print(f"\n═══ Phase 6: Plots → {output_dir} ═══")

    m_star, m_p, R_p, R_star = _derive_physics(cfg)
    viz = cfg.visualization
    dpi = max(150, int(viz.figure_dpi))
    landscape_size = (14.0, 8.0)
    saved: List[str] = []

    def _save(fig, name):
        p = output_dir / name
        fig.set_size_inches(*landscape_size, forward=True)
        fig.savefig(p, dpi=dpi)
        saved.append(str(p))
        plt.close(fig)
        if verbose:
            print(f"  ✓ {name}")

    def _save_fig_dict(figs: Dict[str, plt.Figure]):
        for fname, fig in figs.items():
            _save(fig, fname)

    def _capture_new_pngs(before: set[str]):
        after = {p.name for p in output_dir.glob("*.png")}
        new_names = sorted(after - before)
        for n in new_names:
            saved.append(str(output_dir / n))
        return new_names

    # MC summary (standalone panels)
    _save_fig_dict(plot_mc_summary_individual(mc, figsize=landscape_size))

    # Sampling-parameter diagnostics (config proposal-space coverage + cutoffs)
    param_figs = plot_sampling_parameter_distributions(mc, cfg=cfg, save_dir=None, dpi=dpi)
    for fname, fig in param_figs.items():
        _save(fig, fname)
    if verbose and not param_figs:
        print("  ! sampling parameter distributions unavailable")

    # Rejection breakdown
    _save(plot_rejection_breakdown(mc, save_dir=None, dpi=dpi), "rejection_breakdown.png")

    # Parameter correlations (standalone panels)
    _save_fig_dict(plot_parameter_correlations_individual(mc, figsize=landscape_size))

    # Star proximity (standalone panels)
    star_clearance = cfg.numerical.star_min_clearance_Rstar
    _save_fig_dict(plot_star_proximity_distribution_individual(
        mc, R_star, clearance_Rstar=star_clearance, figsize=landscape_size,
    ))

    # Energy CDF — enriched with narrowed baselines + re-run overlay
    cdf_kwargs: Dict[str, Any] = {}
    if baselines:
        narrowed_bl = baselines.get("narrowed")
        if narrowed_bl:
            if narrowed_bl.get("star"):
                cdf_kwargs["E_star_narrowed"] = narrowed_bl["star"].max_energy_half_dv_vec_sq
            if narrowed_bl.get("planet"):
                cdf_kwargs["E_planet_narrowed"] = narrowed_bl["planet"].max_energy_half_dv_vec_sq
    analyses_best_list = rerun.get("analyses_best", [])
    if analyses_best_list:
        cdf_kwargs["analyses_best"] = analyses_best_list
    best_vec_ana = best.get("best_vec_ana")
    if best_vec_ana:
        cdf_kwargs["E_3body_best"] = best_vec_ana.get("energy_half_dv_vec_sq", None)
    cdf_kwargs["system_name"] = cfg.system.name
    _save(plot_energy_cdf(mc, save_dir=None, dpi=dpi, **cdf_kwargs), "energy_cdf.png")

    # Best candidate trajectory
    best_sol = best.get("best_sol")
    best_ana = best.get("best_ana")
    if best_sol is not None:
        _save(plot_best_candidate_with_bodies(
            best_sol, best_ana, m_star=m_star, m_p=m_p, R_p=R_p,
        ), "best_candidate.png")
        _save_fig_dict(plot_velocity_phase_space_individual(
            best_sol, title_prefix="Best", figsize=landscape_size,
        ))

    # Planet-frame diagnostics
    analyses = rerun["analyses_best"]
    valid_ana = [a for a in analyses if a is not None]
    if valid_ana:
        _save_fig_dict(plot_planet_frame_diagnostics_individual(
            valid_ana, R_p, R_star, figsize=landscape_size,
        ))

    # Publication objectives dashboard
    if viz.generate_publication_dashboard:
        _save_fig_dict(plot_publication_objectives_individual(
            mc=mc,
            analyses_best=valid_ana if valid_ana else None,
            comparison=baselines.get("comparison") if baselines else None,
            R_star_km=R_star,
            clearance_Rstar=cfg.numerical.star_min_clearance_Rstar,
            figsize=landscape_size,
        ))

    # Candidate ranking diagnostics
    if viz.generate_candidate_ranking_plot:
        _save_fig_dict(plot_candidate_ranking_diagnostics_individual(
            analyses=analyses,
            top_indices=top_idx,
            R_p_km=R_p,
            R_star_km=R_star,
            figsize=landscape_size,
        ))

    # Multi-candidate overlay
    sols = rerun["sols_best"]
    top_n = cfg.visualization.top_n_overlay
    if sols:
        _save(plot_multi_candidate_overlay(
            sols, analyses, m_star=m_star, m_p=m_p,
            R_star=R_star, R_p=R_p, top_n=top_n, save_dir=None, dpi=dpi,
        ), "multi_candidate_overlay.png")

    # Baseline comparison plot (from baselines module)
    if best_sol is not None:
        enc = extract_encounter_states(best_sol, m_p=m_p, R_p=R_p,
                                       r_far_factor=cfg.numerical.r_far_factor,
                                       min_clearance_factor=cfg.numerical.min_clearance_factor)
        if enc.ok:
            try:
                compare_3body_with_baselines(
                    best_sol, enc,
                    m_star=m_star, m_p=m_p, R_p=R_p,
                    make_plots=True,
                    plot_save_dir=str(output_dir),
                )
                saved.append(str(output_dir / "baseline_comparison.png"))
                if verbose:
                    print(f"  ✓ baseline_comparison.png")
            except Exception as e:
                if verbose:
                    print(f"  ✗ baseline_comparison: {e}")

    # 2-body parameter-space heatmaps (from plotting_twobody)
    try:
        from .plotting_twobody import (
            plot_poincare_heatmaps,
            plot_scattering_maps,
            plot_encounter_2d_cartesian,
            plot_encounter_2d_trajectories,
            plot_oberth_comparison,
        )
        res = viz.heatmap_grid_resolution
        angles = viz.heatmap_approach_angles_deg

        tb = cfg.two_body
        v_app = tb.v_approach_kms if tb else 50.0
        narrowed_env = None
        if baselines and baselines.get("narrowed") is not None:
            narrowed_env = baselines["narrowed"].get("envelope")
        if narrowed_env is not None and hasattr(narrowed_env, "vstar_vec"):
            vstar0 = (float(narrowed_env.vstar_vec[0]), float(narrowed_env.vstar_vec[1]))
            if verbose:
                print(f"  2-body vstar sync: using narrowed-envelope value ({vstar0[0]:.3f}, {vstar0[1]:.3f}) km/s")
        else:
            vstar0 = _resolve_two_body_star_velocity(cfg, mc=mc, verbose=verbose)
        b_min = tb.b_min_km if tb else 1e7
        b_max = tb.b_max_km if tb else 4e9

        if viz.generate_poincare_maps:
            before_png = {p.name for p in output_dir.glob("*.png")}
            figs = plot_poincare_heatmaps(
                m_star, v_inf_kms=v_app, vstar0_kms=vstar0,
                num_b=res, num_angle=res,
                b_min_km=b_min, b_max_km=b_max,
                body_label=cfg.system.name + " Star",
                save_dir=output_dir, dpi=dpi,
            )
            for f in figs:
                plt.close(f)
            _capture_new_pngs(before_png)
            if verbose:
                print(f"  ✓ poincare_heatmaps ({len(figs)} figs)")

        if viz.generate_scattering_maps:
            before_png = {p.name for p in output_dir.glob("*.png")}
            figs = plot_scattering_maps(
                m_star, v_approach_kms=v_app, vstar0_kms=vstar0,
                approach_angles_deg=angles, num_b=res, num_theta=res,
                b_min_km=b_min, b_max_km=b_max,
                body_label=cfg.system.name,
                save_dir=output_dir, dpi=dpi,
            )
            for f in figs:
                plt.close(f)
            _capture_new_pngs(before_png)
            if verbose:
                print(f"  ✓ scattering_maps ({len(figs)} figs)")

        if viz.generate_2body_heatmaps:
            before_png = {p.name for p in output_dir.glob("*.png")}
            figs = plot_encounter_2d_cartesian(
                m_star, v_approach_kms=v_app, vstar0_kms=vstar0,
                approach_angles_deg=angles, num_xy=res,
                body_label=cfg.system.name,
                save_dir=output_dir, dpi=dpi,
            )
            for f in figs:
                plt.close(f)
            _capture_new_pngs(before_png)
            if verbose:
                print(f"  ✓ encounter_2d_cartesian ({len(figs)} figs)")

        if viz.generate_oberth_maps:
            before_png = {p.name for p in output_dir.glob("*.png")}
            figs = plot_oberth_comparison(
                m_star, v_inf_kms=v_app, vstar0_kms=vstar0,
                num_b=res, num_angle=res,
                b_min_km=b_min, b_max_km=b_max,
                body_label=cfg.system.name,
                save_dir=output_dir, dpi=dpi,
            )
            for f in figs:
                plt.close(f)
            _capture_new_pngs(before_png)
            if verbose:
                print(f"  ✓ oberth_comparison ({len(figs)} figs)")

    except Exception as e:
        if verbose:
            print(f"  ✗ 2-body heatmaps skipped: {e}")

    # Trajectory tracks (requires narrowed baselines)
    try:
        from .plotting_twobody import plot_trajectory_tracks
        narrowed = baselines.get("narrowed") if baselines else None
        if narrowed is not None and narrowed.get("envelope") is not None:
            viz = cfg.visualization
            norm_mode = getattr(viz, "trajectory_energy_norm_mode", "auto")
            fixed_range = None
            if norm_mode == "fixed":
                vmin = getattr(viz, "trajectory_energy_vmin", None)
                vmax = getattr(viz, "trajectory_energy_vmax", None)
                if vmin is not None and vmax is not None and vmax > vmin:
                    fixed_range = (float(vmin), float(vmax))
            before_png = {p.name for p in output_dir.glob("*.png")}
            figs_tt = plot_trajectory_tracks(
                narrowed=narrowed,
                sols_best=rerun.get("sols_best", []),
                analyses_best=rerun.get("analyses_best", []),
                cfg=cfg,
                overlay_lines=getattr(viz, "trajectory_overlay_lines", True),
                overlay_line_count=getattr(viz, "trajectory_overlay_line_count", 90),
                gradient_mode=getattr(viz, "trajectory_gradient_mode", "hexbin"),
                confidence_min_count=getattr(viz, "trajectory_confidence_min_count", 2),
                fixed_energy_range=fixed_range,
                hexbin_gridsize=getattr(viz, "trajectory_hexbin_gridsize", 150),
                kde_sigma_bins=getattr(viz, "trajectory_kde_sigma_bins", 2.0),
                time_frames=getattr(viz, "trajectory_time_frames", 48),
                export_time_data=getattr(viz, "trajectory_time_export_npz", True),
                save_dir=str(output_dir),
                dpi=dpi,
            )
            for fig_tt in figs_tt:
                plt.close(fig_tt)
            _capture_new_pngs(before_png)
            if verbose:
                print(f"  ✓ trajectory_tracks ({len(figs_tt)} figs)")
    except Exception as e:
        if verbose:
            print(f"  ✗ trajectory_tracks skipped: {e}")

    return saved


def phase_animations(
    cfg: FullConfig,
    best_sol,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, str]:
    """Phase 7: Generate animations / videos."""
    if not cfg.visualization.render_video:
        if verbose:
            print("\n═══ Phase 7: Animations — SKIPPED (disabled) ═══")
        return {}

    if verbose:
        print(f"\n═══ Phase 7: Animations ═══")

    if best_sol is None:
        if verbose:
            print("  No valid solution for animation")
        return {}

    frames_dir = str(output_dir / "frames")
    animations = generate_all_animations(
        best_sol,
        output_dir=frames_dir,
        video_fps=cfg.visualization.video_fps,
        video_format=cfg.visualization.video_format,
        animate_trajectory=cfg.visualization.animate_trajectory,
        animate_phase_space=cfg.visualization.animate_phase_space,
    )

    if verbose:
        for atype, fpath in animations.items():
            status = "✓" if fpath else "✗"
            print(f"  {status} {atype}: {fpath or 'failed'}")

    return animations


def phase_save(
    cfg: FullConfig,
    mc: Dict[str, Any],
    top_idx: np.ndarray,
    rerun: Dict[str, Any],
    best: Dict[str, Any],
    baselines: Dict[str, Any],
    output_dir: Path,
    verbose: bool = True,
) -> None:
    """Phase 8: Save config, results pickle, CSV summary."""
    if verbose:
        print(f"\n═══ Phase 8: Save → {output_dir} ═══")

    # Config
    save_config(cfg, str(output_dir / "config.yaml"), format="yaml")
    if verbose:
        print(f"  ✓ config.yaml")

    # Results pickle
    try:
        results_pkl = output_dir / "results.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(_strip_dataclasses({
                "mc": mc,
                "top_indices": top_idx,
                "sols_best": rerun["sols_best"],
                "analyses_best": rerun["analyses_best"],
                "comparison": baselines.get("comparison"),
            }), f)
        if verbose:
            print(f"  ✓ results.pkl")
    except Exception as e:
        if verbose:
            print(f"  ✗ results.pkl: {e}")

    # CSV summary
    try:
        import pandas as pd
        rows = []
        analyses = rerun["analyses_best"]
        for i, (idx, ana) in enumerate(zip(top_idx, analyses)):
            if ana is not None:
                enc = ana.get("encounter")
                r_star = enc.r_star_min if enc and enc.r_star_min else float("nan")
                rows.append({
                    "Rank": i + 1,
                    "MC_idx": int(idx),
                    "dv_kms": round(ana["delta_v"], 4),
                    "dv_pct": round(ana["delta_v_pct"], 2),
                    "dv_vec_kms": round(ana.get("delta_v_vec", 0), 4),
                    "half_dv_vec_sq": round(ana.get("energy_half_dv_vec_sq", 0), 4),
                    "deflection_deg": round(ana["deflection"], 1),
                    "r_min_planet_km": round(ana["r_min"], 0),
                    "r_min_star_km": round(r_star, 0),
                })
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary.csv", index=False)
        if verbose:
            print(f"  ✓ summary.csv ({len(rows)} candidates)")
    except Exception as e:
        if verbose:
            print(f"  ✗ summary.csv: {e}")


# ===================================================================
# Master entrypoint
# ===================================================================

def run_pipeline(
    config_path: str,
    output_dir: Optional[str] = None,
    phases: Optional[List[str]] = None,
    skip_plots: bool = False,
    skip_animations: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full slingshot pipeline: config in → results out.

    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    output_dir : str, optional
        Override output directory. Auto-generated if None.
    phases : list of str, optional
        Subset of phases to run.  If None, runs all.
        Valid: ``"mc"``, ``"select"``, ``"rerun"``, ``"best"``,
               ``"baselines"``, ``"plots"``, ``"animations"``, ``"save"``.
    skip_plots : bool
        Skip all plot generation.
    skip_animations : bool
        Skip animation / video generation.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        All intermediate results keyed by phase name.
    """
    # Ensure nested module prints cannot crash on restrictive console encodings.
    configure_console_streams()

    all_phases = {"mc", "select", "rerun", "best", "baselines",
                  "plots", "animations", "save"}
    if phases is None:
        run_phases = all_phases
    else:
        run_phases = set(phases)
        unknown = run_phases - all_phases
        if unknown:
            raise ValueError(f"Unknown phases: {unknown}. Valid: {sorted(all_phases)}")

    if skip_plots:
        run_phases.discard("plots")
    if skip_animations:
        run_phases.discard("animations")

    # Load config
    cfg = load_config(config_path)
    if verbose:
        print(f"╔══════════════════════════════════════════════╗")
        print(f"║  Slingshot Pipeline — {cfg.system.name:>20s}  ║")
        print(f"╚══════════════════════════════════════════════╝")
        print(f"  Config: {config_path}")

    # Output directory
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(f"results/results_{cfg.system.name}_{ts}")
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  Output: {out}\n")

    results: Dict[str, Any] = {"config": cfg, "output_dir": out}

    # Phase 1 — MC
    if "mc" in run_phases:
        mc = phase_monte_carlo(cfg, verbose=verbose)
        results["mc"] = mc
    else:
        mc = results.get("mc")

    if mc is None:
        if verbose:
            print("  No MC data — pipeline stopped.")
        return results

    # Phase 2 — Selection
    if "select" in run_phases:
        top_idx = phase_select(cfg, mc, verbose=verbose)
        results["top_idx"] = top_idx
    else:
        top_idx = results.get("top_idx")

    if top_idx is None or len(top_idx) == 0:
        if verbose:
            print("  No candidates selected — pipeline stopped.")
        return results

    # Phase 3 — Re-run
    if "rerun" in run_phases:
        rerun = phase_rerun(cfg, mc, top_idx, verbose=verbose)
        results["rerun"] = rerun
    else:
        rerun = results.get("rerun")

    if rerun is None:
        if verbose:
            print("  No re-run data — pipeline stopped.")
        return results

    # Phase 4 — Best selection
    if "best" in run_phases:
        best = phase_best_selection(
            rerun["analyses_best"], top_idx, rerun["sols_best"], verbose=verbose,
        )
        results["best"] = best
    else:
        best = results.get("best", {})

    # Phase 5 — Baselines
    if "baselines" in run_phases:
        baselines = phase_baselines(
            cfg, rerun["analyses_best"],
            best.get("best_vec_ana"), verbose=verbose,
        )
        results["baselines"] = baselines
    else:
        baselines = results.get("baselines", {})

    # Phase 6 — Plots
    if "plots" in run_phases:
        saved_plots = phase_plots(
            cfg, mc, top_idx, rerun, best, out,
            verbose=verbose, baselines=baselines,
        )
        results["saved_plots"] = saved_plots

    # Phase 7 — Animations
    if "animations" in run_phases:
        anims = phase_animations(cfg, best.get("best_sol"), out, verbose=verbose)
        results["animations"] = anims

    # Phase 8 — Save
    if "save" in run_phases:
        phase_save(cfg, mc, top_idx, rerun, best, baselines, out, verbose=verbose)

    # Auto-generate report
    try:
        from .report import generate_run_report
        report = generate_run_report(
            output_dir=out,
            cfg=cfg,
            mc=mc,
            analyses_best=rerun["analyses_best"],
            best=best,
            comparison=baselines.get("comparison") if baselines else None,
            narrowed=baselines.get("narrowed") if baselines else None,
            saved_plots=results.get("saved_plots"),
            top_indices=top_idx,
        )
        results["report"] = report
        if verbose:
            print(f"  ✓ REPORT.md")
    except Exception as e:
        if verbose:
            print(f"  ✗ REPORT.md: {e}")

    if verbose:
        print(f"\n{'═' * 50}")
        print(f"  Pipeline complete → {out}")
        print(f"{'═' * 50}")

    return results
