"""
Unified Monte Carlo particle batch processing.
Handles both barycentric and planet-frame sampling/analysis.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .dynamics import init_hot_jupiter_barycentric, simulate_3body
from .analysis import analyze_trajectory
from .sampling import sample_satellite_state_barycentric, sample_satellite_state_near_planet
from .constants import M_SUN, M_JUP, R_JUP, R_SUN


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
    
    # Initialize star+planet system
    Y_sp0 = init_hot_jupiter_barycentric(m_star=m_star, m_p=m_p)

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
    if sampling_mode == "barycentric":
        sat_states = sample_satellite_state_barycentric(
            Y_sp0, N=N,
            v_mag_min=sampling_kwargs.get('v_mag_min', 10.0),
            v_mag_max=sampling_kwargs.get('v_mag_max', 120.0),
            impact_param_min_AU=sampling_kwargs.get('impact_param_min_AU', 0.5),
            impact_param_max_AU=sampling_kwargs.get('impact_param_max_AU', 3.0),
            angle_in_min_deg=sampling_kwargs.get('angle_in_min_deg', -60.0),
            angle_in_max_deg=sampling_kwargs.get('angle_in_max_deg', 60.0),
            r_init_AU=sampling_kwargs.get('r_init_AU', None),
            rng=rng,
        )
    elif sampling_mode == "planet":
        sat_states = sample_satellite_state_near_planet(
            Y_sp0, N=N,
            R_p=R_p,
            r_min_factor=sampling_kwargs.get('r_min_factor', 20.0),
            r_max_factor=sampling_kwargs.get('r_max_factor', 500.0),
            v_rel_min=sampling_kwargs.get('v_rel_min', 12.0),
            v_rel_max=sampling_kwargs.get('v_rel_max', 80.0),
            rng=rng,
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
            deflection[idx] = ana["deflection"]
            delta_v_vec[idx] = ana.get("delta_v_vec", np.nan)
            energy_half_dv_vec_sq[idx] = ana.get("energy_half_dv_vec_sq", np.nan)
    
    ok_count = ok_flags.sum()
    if verbose:
        print(f"Results: {ok_count}/{N} successful ({100.0*ok_count/N:.1f}%)")
    
    return {
        "Y_sp0": Y_sp0,
        "sat_states": sat_states,
        "ok": ok_flags,
        "delta_v": delta_v,
        "deflection": deflection,
        "r_min": r_min_arr,
        "delta_v_vec": delta_v_vec,
        "energy_half_dv_vec_sq": energy_half_dv_vec_sq,
        "r_star_min": r_star_min_arr,
        "results": results_list,
        "m_star": m_star,
        "m_p": m_p,
        "R_p": R_p,
        "r_hill": r_hill,
        "frame": frame,
        "sampling_mode": sampling_mode,
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
    ok = mc["ok"]
    ok_idx = np.where(ok)[0]
    
    if ok_idx.size == 0:
        return np.array([], dtype=int)
    
    # Get metric values
    metric_vals = mc[metric][ok_idx]
    
    # Sort
    if sign == "maximize":
        order = np.argsort(metric_vals)[::-1]
    elif sign == "minimize":
        order = np.argsort(metric_vals)
    elif sign == "abs":
        order = np.argsort(np.abs(metric_vals))[::-1]
    else:
        order = np.argsort(metric_vals)[::-1]
    
    n_top = max(min_top, int(np.ceil(top_frac * ok_idx.size)))
    top_local = order[:n_top]
    
    return ok_idx[top_local]
