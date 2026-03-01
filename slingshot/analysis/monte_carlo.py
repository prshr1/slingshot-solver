"""
Unified Monte Carlo particle batch processing.
Handles both barycentric and planet-frame sampling/analysis.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.dynamics import init_hot_jupiter_barycentric, simulate_3body
from .trajectory import analyze_trajectory
from ..core.sampling import sample_satellite_state_barycentric, sample_satellite_state_near_planet
from ..constants import M_SUN, M_JUP, R_JUP, R_SUN


_METRIC_ALIASES = {
    # Legacy aliases used by historical configs
    "bary_delta_v": "delta_v",
    "bary_delta_v_pct": "delta_v_pct",
    "bary_delta_v_abs": "delta_v_abs",
    "planet_delta_v": "delta_v_planet_frame",
    # Canonical shorthand aliases
    "dv": "delta_v",
    "dv_pct": "delta_v_pct",
    "dv_vec": "delta_v_vec",
    "half_dv_vec_sq": "energy_half_dv_vec_sq",
}


def _resolve_metric_name(metric: str) -> str:
    """Resolve user/config metric names to canonical MC array keys."""
    key = str(metric).strip().lower()
    return _METRIC_ALIASES.get(key, key)


def resolve_metric_array(mc: Dict[str, Any], metric: str) -> np.ndarray:
    """Return a full-length metric array from MC results (derived if needed)."""
    canonical = _resolve_metric_name(metric)

    if canonical in mc:
        return np.asarray(mc[canonical], dtype=float)

    if canonical == "delta_v_abs":
        return np.abs(np.asarray(mc["delta_v"], dtype=float))
    if canonical == "deflection_abs":
        return np.abs(np.asarray(mc["deflection"], dtype=float))

    raise KeyError(
        f"Unknown metric '{metric}' (canonical '{canonical}'). "
        f"Available keys: {sorted(k for k in mc.keys() if isinstance(mc[k], np.ndarray))}"
    )


def _apply_objective_sign(vals: np.ndarray, sign: str) -> np.ndarray:
    """Transform objective values so higher is always better."""
    s = str(sign).strip().lower()
    if s == "maximize":
        return vals
    if s == "minimize":
        return -vals
    if s == "abs":
        return np.abs(vals)
    raise ValueError(f"Unknown objective sign: {sign}. Use maximize|minimize|abs")


def _compute_n_top(n_valid: int, top_frac: float, min_top: int) -> int:
    if n_valid <= 0:
        return 0
    return max(min_top, int(np.ceil(top_frac * n_valid)))


def _weighted_scores_from_transformed(
    transformed: np.ndarray,
    weights: Optional[np.ndarray] = None,
    normalization: str = "minmax",
) -> np.ndarray:
    """Compute weighted scalar scores from transformed objective matrix."""
    if transformed.ndim != 2:
        raise ValueError("Expected transformed objective matrix with shape [N, M]")

    if weights is None:
        w = np.ones(transformed.shape[1], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if w.size != transformed.shape[1]:
        raise ValueError("weights length must match number of objectives")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.sum(w) <= 0:
        w = np.ones_like(w)

    mode = str(normalization).strip().lower()
    if mode == "minmax":
        vmin = np.min(transformed, axis=0)
        vmax = np.max(transformed, axis=0)
        span = vmax - vmin
        norm = np.zeros_like(transformed)
        nz = span > 0
        norm[:, nz] = (transformed[:, nz] - vmin[nz]) / span[nz]
        # Constant objectives carry no ranking power -> neutral midpoint.
        norm[:, ~nz] = 0.5
    elif mode == "rank":
        # Rank-normalize each objective to [0,1].
        norm = np.zeros_like(transformed)
        n = transformed.shape[0]
        if n == 1:
            norm[:] = 1.0
        else:
            for j in range(transformed.shape[1]):
                order = np.argsort(transformed[:, j])
                r = np.empty_like(order, dtype=float)
                r[order] = np.arange(n, dtype=float)
                norm[:, j] = r / (n - 1)
    else:
        raise ValueError(f"Unknown weighted normalization: {normalization}")

    return (norm @ w) / np.sum(w)


def _prepare_objective_matrix(
    mc: Dict[str, Any],
    objectives: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build objective matrix for successful particles.

    Returns
    -------
    tuple
        ok_idx_valid, transformed_values, weights
    """
    ok_idx_all = np.where(mc["ok"])[0]
    if ok_idx_all.size == 0:
        return (
            np.array([], dtype=int),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
        )

    cols: List[np.ndarray] = []
    weights: List[float] = []

    for obj in objectives:
        metric = obj.get("metric", "delta_v")
        sign = obj.get("sign", "maximize")
        weight = float(obj.get("weight", 1.0))
        vals = resolve_metric_array(mc, metric)[ok_idx_all]
        cols.append(_apply_objective_sign(vals, sign))
        weights.append(weight)

    transformed = np.column_stack(cols) if cols else np.empty((ok_idx_all.size, 0), dtype=float)

    if transformed.size == 0:
        return (
            np.array([], dtype=int),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
        )

    finite_mask = np.all(np.isfinite(transformed), axis=1)
    return ok_idx_all[finite_mask], transformed[finite_mask], np.asarray(weights, dtype=float)


def _pareto_front_mask(values: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated points for max-oriented objectives."""
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


def evaluate_particle(
    particle_args: Tuple,
) -> Dict[str, Any]:
    """
    Evaluate single particle trajectory. Designed for parallel execution.
    
    Parameters
    ----------
    particle_args : tuple
        (idx, sat_state, Y_sp0, t_span, m_star, m_p, R_p, frame, **ana_kwargs)
    
    Returns
    -------
    dict
        {"idx": idx, "ok": bool, "reason": str, "analysis": dict (if ok), "sol": sol}
    """
    idx, sat_state, Y_sp0, t_span, m_star, m_p, R_p, frame, ana_kwargs = particle_args
    
    # Build full state
    xs, ys, vxs, vys, xp, yp, vxp, vyp = Y_sp0
    x3, y3, vx3, vy3 = sat_state
    
    Y0 = np.array([xs, ys, vxs, vys, xp, yp, vxp, vyp, x3, y3, vx3, vy3], dtype=float)
    
    # Escape radius: use initial barycentric distance × escape_radius_factor
    # so integration continues until the particle has fully completed its swing
    r0_bary = np.sqrt(x3 * x3 + y3 * y3)
    escape_factor = ana_kwargs.get('escape_radius_factor', 3.0)
    escape_r = ana_kwargs.get('escape_radius_km', r0_bary * escape_factor)
    
    # Integrate
    sol = simulate_3body(
        Y0, t_span,
        m_star=m_star,
        m_p=m_p,
        rtol=ana_kwargs.get('rtol', 1e-10),
        atol=ana_kwargs.get('atol', 1e-10),
        escape_radius_km=escape_r,
        method=ana_kwargs.get('ode_method', 'DOP853'),
        softening_km=ana_kwargs.get('softening_km', 0.0),
    )
    
    if sol is None:
        return {
            "idx": idx,
            "ok": False,
            "reason": "integration_failed",
            "analysis": None,
            "sol": None,
        }
    
    # Analyze
    analysis = analyze_trajectory(
        sol,
        frame=frame,
        m_star=m_star,
        m_p=m_p,
        R_p=R_p,
        r_far_factor=ana_kwargs.get('r_far_factor', 20.0),
        min_clearance_factor=ana_kwargs.get('min_clearance_factor', 1.05),
    )
    
    if analysis is None:
        return {
            "idx": idx,
            "ok": False,
            "reason": "analysis_failed",
            "analysis": None,
            "sol": sol,
        }
    
    # Apply filtering
    if frame == "planet":
        if not analysis["unbound_f"]:
            return {
                "idx": idx,
                "ok": False,
                "reason": "not_unbound_planet_frame",
                "analysis": analysis,
                "sol": sol,
            }
        if analysis["delta_v"] <= 0:
            return {
                "idx": idx,
                "ok": False,
                "reason": "no_energy_gain",
                "analysis": analysis,
                "sol": sol,
            }
    
    elif frame == "barycentric":
        # Apply barycentric filtering if requested
        bary_req = ana_kwargs.get('bary_unbound_requirement', None)
        if bary_req is not None:
            unbound_i = analysis.get("unbound_i", False)
            unbound_f = analysis.get("unbound_f", False)
            
            pass_bary = False
            if bary_req == 'pre':
                pass_bary = unbound_i
            elif bary_req == 'post':
                pass_bary = unbound_f
            elif bary_req == 'either':
                pass_bary = (unbound_i or unbound_f)
            elif bary_req == 'both':
                pass_bary = (unbound_i and unbound_f)
            
            if not pass_bary:
                return {
                    "idx": idx,
                    "ok": False,
                    "reason": "bary_unbound_requirement_failed",
                    "analysis": analysis,
                    "sol": sol,
                }
    
    # --- Flyby completion filter ---
    enc = analysis.get("encounter", None)
    if enc is not None:
        # (a) Temporal ordering: particle must go inward then come back out
        if enc.i0 is not None and enc.k_min is not None and enc.i1 is not None:
            if not (enc.i0 < enc.k_min < enc.i1):
                return {
                    "idx": idx,
                    "ok": False,
                    "reason": "flyby_incomplete",
                    "analysis": analysis,
                    "sol": sol,
                }
        else:
            # Missing encounter indices → incomplete
            return {
                "idx": idx,
                "ok": False,
                "reason": "flyby_incomplete",
                "analysis": analysis,
                "sol": sol,
            }

        # (b) Max closest-approach distance (in km, pre-computed from Hill radii)
        flyby_r_max = ana_kwargs.get('flyby_r_min_max_km', None)
        if flyby_r_max is not None and enc.r_min is not None:
            if enc.r_min > flyby_r_max:
                return {
                    "idx": idx,
                    "ok": False,
                    "reason": "flyby_too_distant",
                    "analysis": analysis,
                    "sol": sol,
                }

        # (c) Star proximity filter — reject star-penetrating trajectories
        star_min_clearance_km = ana_kwargs.get('star_min_clearance_km', None)
        if star_min_clearance_km is not None and enc.r_star_min is not None:
            if enc.r_star_min < star_min_clearance_km:
                return {
                    "idx": idx,
                    "ok": False,
                    "reason": "star_too_close",
                    "analysis": analysis,
                    "sol": sol,
                }

    return {
        "idx": idx,
        "ok": True,
        "reason": "",
        "analysis": analysis,
        "sol": sol,
    }


def run_monte_carlo(
    N: int = 3000,
    t_span: Tuple[float, float] = (0.0, 1e7),
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    R_p: Optional[float] = None,
    frame: str = "barycentric",
    sampling_mode: str = "barycentric",
    n_parallel: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    **sampling_kwargs,
) -> Dict[str, Any]:
    """
    Run unified Monte Carlo sweep with configurable sampling and analysis.
    
    Parameters
    ----------
    N : int
        Number of test particles
    t_span : tuple
        (t_start, t_end) integration time span
    m_star : float, optional
        Star mass. Defaults to 1 solar mass.
    m_p : float, optional
        Planet mass. Defaults to 1 Jupiter mass.
    R_p : float, optional
        Planet radius. Defaults to 1 Jupiter radius.
    frame : str
        "planet" or "barycentric" for analysis frame
    sampling_mode : str
        "barycentric" or "planet" for sampling mode
    n_parallel : int, optional
        Number of parallel workers. If None, runs serially. If >0, uses ProcessPoolExecutor.
    rng : np.random.Generator, optional
        Random number generator
    verbose : bool
        Print progress messages
    **sampling_kwargs
        Additional kwargs passed to sampling/analysis functions
    
    Returns
    -------
    dict
        Results with keys: Y_sp0, sat_states, ok, individual result metrics
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    if R_p is None:
        R_p = R_JUP
    if rng is None:
        rng = np.random.default_rng()
    
    if verbose:
        print(f"Monte Carlo: {N} particles, {frame} frame, {sampling_mode} sampling")
    
    # Initialize star+planet system (optionally with system bulk velocity)
    bulk_vx = sampling_kwargs.get('bulk_velocity_vx_kms', 0.0)
    bulk_vy = sampling_kwargs.get('bulk_velocity_vy_kms', 0.0)
    Y_sp0 = init_hot_jupiter_barycentric(
        m_star=m_star,
        m_p=m_p,
        bulk_velocity_vx_kms=bulk_vx,
        bulk_velocity_vy_kms=bulk_vy,
    )
    if verbose and (bulk_vx != 0.0 or bulk_vy != 0.0):
        v_bulk = np.hypot(bulk_vx, bulk_vy)
        print(f"  System bulk velocity: ({bulk_vx:.2f}, {bulk_vy:.2f}) km/s |v|={v_bulk:.2f} km/s")

    # Compute Hill sphere and flyby distance threshold
    xs, ys = Y_sp0[0], Y_sp0[1]
    xp, yp = Y_sp0[4], Y_sp0[5]
    a_km = np.sqrt((xp - xs)**2 + (yp - ys)**2)
    r_hill = a_km * (m_p / (3.0 * m_star)) ** (1.0 / 3.0)
    flyby_r_hill_factor = sampling_kwargs.get('flyby_r_min_max_hill', None)
    if flyby_r_hill_factor is not None:
        flyby_r_min_max_km = flyby_r_hill_factor * r_hill
        sampling_kwargs['flyby_r_min_max_km'] = flyby_r_min_max_km
        if verbose:
            print(f"  Hill sphere: {r_hill:.0f} km ({r_hill/1.496e8:.4f} AU)")
            print(f"  Flyby r_min threshold: {flyby_r_min_max_km:.0f} km "
                  f"({flyby_r_hill_factor:.1f} × r_Hill)")

    # Star proximity filter — pre-compute threshold in km
    star_min_clearance_Rstar = sampling_kwargs.get('star_min_clearance_Rstar', None)
    if star_min_clearance_Rstar is not None:
        R_star_Rsun = sampling_kwargs.get('R_star_Rsun', 1.0)
        R_star_km = R_star_Rsun * R_SUN
        star_min_clearance_km = star_min_clearance_Rstar * R_star_km
        sampling_kwargs['star_min_clearance_km'] = star_min_clearance_km
        if verbose:
            print(f"  Star filter: r_min_star > {star_min_clearance_Rstar:.1f} R★ "
                  f"({star_min_clearance_km:.0f} km)")

    # Sample satellite states
    sampling_params: Dict[str, np.ndarray] = {}
    if sampling_mode == "barycentric":
        sat_states, sampling_params = sample_satellite_state_barycentric(
            Y_sp0, N=N,
            v_mag_min=sampling_kwargs.get('v_mag_min', 10.0),
            v_mag_max=sampling_kwargs.get('v_mag_max', 120.0),
            impact_param_min_AU=sampling_kwargs.get('impact_param_min_AU', 0.5),
            impact_param_max_AU=sampling_kwargs.get('impact_param_max_AU', 3.0),
            angle_in_min_deg=sampling_kwargs.get('angle_in_min_deg', -60.0),
            angle_in_max_deg=sampling_kwargs.get('angle_in_max_deg', 60.0),
            r_init_AU=sampling_kwargs.get('r_init_AU', None),
            rng=rng,
            return_metadata=True,
        )
    elif sampling_mode == "planet":
        sat_states, sampling_params = sample_satellite_state_near_planet(
            Y_sp0, N=N,
            R_p=R_p,
            r_min_factor=sampling_kwargs.get('r_min_factor', 20.0),
            r_max_factor=sampling_kwargs.get('r_max_factor', 500.0),
            v_rel_min=sampling_kwargs.get('v_rel_min', 12.0),
            v_rel_max=sampling_kwargs.get('v_rel_max', 80.0),
            rng=rng,
            return_metadata=True,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    
    # Prepare particle arguments for evaluation
    particle_args_list = [
        (
            i,
            sat_states[i],
            Y_sp0,
            t_span,
            m_star,
            m_p,
            R_p,
            frame,
            sampling_kwargs,
        )
        for i in range(N)
    ]
    
    # Run evaluations
    results_list = []
    
    if n_parallel and n_parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(evaluate_particle, args): i for i, args in enumerate(particle_args_list)}
            
            for future in as_completed(futures):
                result = future.result()
                results_list.append(result)
                
                if verbose and len(results_list) % max(1, N // 10) == 0:
                    print(f"  Completed: {len(results_list)}/{N}")
    else:
        # Serial execution
        for i, args in enumerate(particle_args_list):
            result = evaluate_particle(args)
            results_list.append(result)
            
            if verbose and (i + 1) % max(1, N // 10) == 0:
                print(f"  Completed: {i + 1}/{N}")
    
    # Sort results by index to match input order
    results_list.sort(key=lambda r: r["idx"])
    
    # Aggregate results
    ok_flags = np.zeros(N, dtype=bool)
    delta_v = np.full(N, np.nan)
    deflection = np.full(N, np.nan)
    r_min_arr = np.full(N, np.nan)
    delta_v_vec = np.full(N, np.nan)
    energy_half_dv_vec_sq = np.full(N, np.nan)
    r_star_min_arr = np.full(N, np.nan)
    delta_v_pct = np.full(N, np.nan)
    delta_v_planet_frame = np.full(N, np.nan)
    energy_from_planet_orbit = np.full(N, np.nan)
    
    for result in results_list:
        idx = result["idx"]
        ok_flags[idx] = result["ok"]
        
        if result["analysis"]:
            ana = result["analysis"]
            r_min_arr[idx] = ana.get("r_min", np.nan)
            enc_r = ana.get("encounter", None)
            if enc_r is not None and enc_r.r_star_min is not None:
                r_star_min_arr[idx] = enc_r.r_star_min
        
        if result["ok"] and result["analysis"]:
            ana = result["analysis"]
            delta_v[idx] = ana["delta_v"]
            delta_v_pct[idx] = ana.get("delta_v_pct", np.nan)
            deflection[idx] = ana["deflection"]
            delta_v_vec[idx] = ana.get("delta_v_vec", np.nan)
            energy_half_dv_vec_sq[idx] = ana.get("energy_half_dv_vec_sq", np.nan)
            delta_v_planet_frame[idx] = ana.get("delta_v_planet_frame", np.nan)
            energy_from_planet_orbit[idx] = ana.get("energy_from_planet_orbit", np.nan)
    
    ok_count = ok_flags.sum()
    if verbose:
        print(f"Results: {ok_count}/{N} successful ({100.0*ok_count/N:.1f}%)")
    
    return {
        "Y_sp0": Y_sp0,
        "sat_states": sat_states,
        "ok": ok_flags,
        "delta_v": delta_v,
        "delta_v_pct": delta_v_pct,
        "deflection": deflection,
        "r_min": r_min_arr,
        "delta_v_vec": delta_v_vec,
        "energy_half_dv_vec_sq": energy_half_dv_vec_sq,
        "delta_v_planet_frame": delta_v_planet_frame,
        "energy_from_planet_orbit": energy_from_planet_orbit,
        "r_star_min": r_star_min_arr,
        "results": results_list,
        "m_star": m_star,
        "m_p": m_p,
        "R_p": R_p,
        "r_hill": r_hill,
        "frame": frame,
        "sampling_mode": sampling_mode,
        "sampling_params": sampling_params,
    }


def select_top_indices(
    mc: Dict[str, Any],
    top_frac: float = 0.10,
    min_top: int = 1,
    metric: str = "delta_v",
    sign: str = "maximize",
) -> np.ndarray:
    """
    Select top N indices from Monte Carlo results.
    
    Parameters
    ----------
    mc : dict
        Monte Carlo results dict from run_monte_carlo
    top_frac : float
        Fraction of successful cases to select
    min_top : int
        Minimum number to select
    metric : str
        Metric to sort by: "delta_v", "deflection", etc.
    sign : str
        "maximize" (largest values), "minimize", or "abs" (largest absolute)
    
    Returns
    -------
    np.ndarray
        Indices of selected particles
    """
    ok_idx_all = np.where(mc["ok"])[0]
    if ok_idx_all.size == 0:
        return np.array([], dtype=int)

    metric_vals_all = resolve_metric_array(mc, metric)[ok_idx_all]
    finite_mask = np.isfinite(metric_vals_all)
    ok_idx = ok_idx_all[finite_mask]
    metric_vals = metric_vals_all[finite_mask]

    if ok_idx.size == 0:
        return np.array([], dtype=int)

    s = str(sign).strip().lower()
    if s == "maximize":
        order = np.argsort(metric_vals)[::-1]
    elif s == "minimize":
        order = np.argsort(metric_vals)
    elif s == "abs":
        order = np.argsort(np.abs(metric_vals))[::-1]
    else:
        raise ValueError(f"Unknown sign: {sign}. Use maximize|minimize|abs")

    n_top = _compute_n_top(ok_idx.size, top_frac, min_top)
    top_local = order[:n_top]
    return ok_idx[top_local]


def select_weighted_indices(
    mc: Dict[str, Any],
    objectives: Sequence[Dict[str, Any]],
    top_frac: float = 0.10,
    min_top: int = 1,
    normalization: str = "minmax",
) -> np.ndarray:
    """Select candidates by weighted normalized objective score."""
    ok_idx, transformed, weights = _prepare_objective_matrix(mc, objectives)
    if ok_idx.size == 0:
        return np.array([], dtype=int)

    n_top = _compute_n_top(ok_idx.size, top_frac, min_top)
    scores = _weighted_scores_from_transformed(
        transformed, weights=weights, normalization=normalization
    )
    order = np.argsort(scores)[::-1]
    return ok_idx[order[:n_top]]


def select_pareto_indices(
    mc: Dict[str, Any],
    objectives: Sequence[Dict[str, Any]],
    top_frac: float = 0.10,
    min_top: int = 1,
    tie_break_normalization: str = "minmax",
) -> np.ndarray:
    """Select candidates via Pareto non-dominated sorting.

    Notes
    -----
    Objectives are transformed so "higher is better" before sorting.
    If the final accepted front exceeds the target size, a weighted
    normalized scalar score is used as deterministic tie-break.
    """
    ok_idx, transformed, weights = _prepare_objective_matrix(mc, objectives)
    if ok_idx.size == 0:
        return np.array([], dtype=int)

    n_top = _compute_n_top(ok_idx.size, top_frac, min_top)
    if n_top >= ok_idx.size:
        return ok_idx

    selected_local: List[int] = []
    remaining_local = np.arange(ok_idx.size, dtype=int)

    while remaining_local.size > 0 and len(selected_local) < n_top:
        vals = transformed[remaining_local]
        front_mask = _pareto_front_mask(vals)
        front_local = remaining_local[front_mask]

        needed = n_top - len(selected_local)
        if front_local.size <= needed:
            selected_local.extend(front_local.tolist())
            remaining_local = remaining_local[~front_mask]
            continue

        front_scores = _weighted_scores_from_transformed(
            transformed[front_local],
            weights=weights,
            normalization=tie_break_normalization,
        )
        front_order = np.argsort(front_scores)[::-1]
        selected_local.extend(front_local[front_order[:needed]].tolist())
        break

    return ok_idx[np.asarray(selected_local, dtype=int)]
