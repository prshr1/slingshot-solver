"""
Narrowed-baseline 2-body comparison for 3-body slingshot results.

Pipeline:
    1. Extract parameter envelope from top 3-body candidates
       (v_approach, impact_parameter, approach_angle ranges).
    2. Pad ranges by a configurable factor.
    3. Run TwoBodyScatter sweeps over the padded envelope
       for both star and planet scattering.
    4. Report max ε and max |ΔV_vec| from each sweep.

This module bridges the gap between the Monte Carlo 3-body solver
and the closed-form TwoBodyScatter engine by ensuring the 2-body
baselines explore the *same* region of parameter space that the
3-body solver actually visited.

Key design choice — vstar0:
    For each candidate's encounter, we extract the star's actual
    barycentric velocity at encounter entry time from the 3-body
    solution.  This gives TwoBodyScatter the exact star-relative
    velocity for a true matched baseline.

Unit contract (matches the rest of slingshot/):
    Distances : km
    Velocities: km/s
    Masses    : kg
    Energy    : km²/s²  (≡ MJ/kg)
    μ         : km³/s²
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .constants import G_KM, M_SUN, M_JUP, R_SUN, R_JUP
from .analysis import EncounterGeometry
from .twobody import TwoBodyEncounter, TrajectoryResult


@dataclass
class EnvelopeParams:
    """Parameter envelope extracted from top 3-body candidates."""

    # Speed ranges (km/s)
    v_approach_min: float
    v_approach_max: float

    # Impact parameter ranges (km)
    b_min: float
    b_max: float

    # Approach angle ranges (radians)
    angle_min: float
    angle_max: float

    # Star velocity estimate (km/s) — median of candidates
    vstar0: float
    vstar_vec: Tuple[float, float]

    # Metadata
    n_candidates: int
    padding_factor: float

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Envelope ({self.n_candidates} candidates, padding={self.padding_factor:.1f}x):\n"
            f"  v_approach : [{self.v_approach_min:.2f}, {self.v_approach_max:.2f}] km/s\n"
            f"  b          : [{self.b_min:.2e}, {self.b_max:.2e}] km\n"
            f"  angle      : [{np.degrees(self.angle_min):.1f}°, {np.degrees(self.angle_max):.1f}°]\n"
            f"  vstar      : ({self.vstar_vec[0]:.2f}, {self.vstar_vec[1]:.2f}) km/s  |v|={self.vstar0:.2f} km/s"
        )


def _extract_candidate_params(
    analysis: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract scattering parameters from one 3-body analysis dict.

    Parameters
    ----------
    analysis : dict
        Output from analyze_trajectory (planet or barycentric frame).

    Returns
    -------
    dict or None
        Keys: v_approach, impact_parameter, angle, vstar0.
        Returns None if encounter data is missing.
    """
    enc_raw = analysis.get("encounter")
    if enc_raw is None:
        return None
    if isinstance(enc_raw, dict):
        if not bool(enc_raw.get("ok", False)):
            return None
        v_rel_i = enc_raw.get("v_rel_i")
        r_rel_i = enc_raw.get("r_rel_i")
        star_v_bary_in = enc_raw.get("star_v_bary_in")
    else:
        enc: Optional[EncounterGeometry] = enc_raw
        if not enc.ok:
            return None
        v_rel_i = enc.v_rel_i
        r_rel_i = enc.r_rel_i
        star_v_bary_in = enc.star_v_bary_in

    # Planet-frame approach speed = |v_rel_i|
    if v_rel_i is not None:
        v_approach = float(np.linalg.norm(v_rel_i))
    else:
        v_approach = analysis.get("v_i", None)
        if v_approach is None:
            return None

    # Impact parameter
    b = analysis.get("impact_parameter", None)
    if b is None and v_rel_i is not None and r_rel_i is not None:
        v_mag = np.linalg.norm(v_rel_i)
        L_mag = abs(np.cross(r_rel_i, v_rel_i))
        b = L_mag / v_mag if v_mag > 0 else 0.0
    if b is None:
        return None

    # Approach angle in planet frame
    if v_rel_i is not None:
        angle = float(np.arctan2(v_rel_i[1], v_rel_i[0]))
    else:
        angle = 0.0

    # Star barycentric speed at encounter entry → vstar0
    vstar_vec = (0.0, 0.0)
    vstar0 = 0.0
    if star_v_bary_in is not None:
        vstar_vec = (float(star_v_bary_in[0]), float(star_v_bary_in[1]))
        vstar0 = float(np.hypot(vstar_vec[0], vstar_vec[1]))

    return {
        "v_approach": v_approach,
        "impact_parameter": b,
        "angle": angle,
        "vstar0": vstar0,
        "vstar_vec": vstar_vec,
    }


def extract_envelope(
    analyses: List[Dict[str, Any]],
    padding_factor: float = 1.5,
) -> Optional[EnvelopeParams]:
    """
    Extract parameter envelope from a list of 3-body analyses.

    Parameters
    ----------
    analyses : list of dict
        Output from analyze_trajectory for each top candidate.
    padding_factor : float
        Multiplicative padding on each range edge.  E.g. 1.5 means
        the envelope is 50% wider on each side of the observed range.

    Returns
    -------
    EnvelopeParams or None
        Padded parameter envelope, or None if no valid candidates.
    """
    params_list = []
    for a in analyses:
        p = _extract_candidate_params(a)
        if p is not None:
            params_list.append(p)

    if not params_list:
        return None

    vs = np.array([p["v_approach"] for p in params_list])
    bs = np.array([p["impact_parameter"] for p in params_list])
    angles = np.array([p["angle"] for p in params_list])
    vstars = np.array([p["vstar0"] for p in params_list])
    vstars_x = np.array([p["vstar_vec"][0] for p in params_list])
    vstars_y = np.array([p["vstar_vec"][1] for p in params_list])

    v_min, v_max = float(vs.min()), float(vs.max())
    b_min, b_max = float(bs.min()), float(bs.max())
    a_min, a_max = float(angles.min()), float(angles.max())

    # Pad ranges symmetrically
    def _pad(lo, hi, factor):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # Ensure minimum spread if all candidates are identical
        if half < 1e-6 * abs(mid):
            half = max(abs(mid) * 0.1, 1.0)
        return mid - half * factor, mid + half * factor

    v_min_p, v_max_p = _pad(v_min, v_max, padding_factor)
    b_min_p, b_max_p = _pad(b_min, b_max, padding_factor)
    a_min_p, a_max_p = _pad(a_min, a_max, padding_factor)

    # Clamp velocity to positive
    v_min_p = max(v_min_p, 0.1)

    # Clamp impact parameter to positive (1 km absolute floor;
    # body-radius enforcement happens in TwoBodyEncounter.compute_trajectory)
    b_min_p = max(b_min_p, 1.0)

    # Star velocity: median vector across candidates
    vstar0_median = float(np.median(vstars))
    vstar_vec_median = (float(np.median(vstars_x)), float(np.median(vstars_y)))
    vstar0_median = float(np.hypot(vstar_vec_median[0], vstar_vec_median[1]))

    return EnvelopeParams(
        v_approach_min=v_min_p,
        v_approach_max=v_max_p,
        b_min=b_min_p,
        b_max=b_max_p,
        angle_min=a_min_p,
        angle_max=a_max_p,
        vstar0=vstar0_median,
        vstar_vec=vstar_vec_median,
        n_candidates=len(params_list),
        padding_factor=padding_factor,
    )


@dataclass
class NarrowedBaselineResult:
    """Results from a narrowed 2-body baseline sweep."""

    label: str                       # "star" or "planet"
    envelope: EnvelopeParams         # Parameter envelope used

    # Best-of-sweep metrics
    max_epsilon: float               # Max specific orbital energy [km²/s²]
    max_deltaV_vec: float            # Max |ΔV_vec| [km/s]
    max_energy_half_dv_vec_sq: float # Max ½ |ΔV_vec|² [km²/s²]

    # Stats
    n_valid: int                     # Number of valid trajectories
    n_total: int                     # Total attempted
    all_epsilons: np.ndarray         # All valid ε values
    all_deltaVs: np.ndarray          # All valid |ΔV_vec| values

    # Sample-level analytics (aligned arrays over valid trajectories)
    sample_v_approach: Optional[np.ndarray] = None   # km/s
    sample_b: Optional[np.ndarray] = None            # km
    sample_angle: Optional[np.ndarray] = None        # rad
    sample_energy: Optional[np.ndarray] = None       # ½|ΔV|² [km²/s²]
    sample_deltaV: Optional[np.ndarray] = None       # |ΔV_vec| [km/s]
    sample_vx_final_rel: Optional[np.ndarray] = None # km/s (relative to moving body)
    sample_vy_final_rel: Optional[np.ndarray] = None # km/s (relative to moving body)

    # Best trajectories (optional — for plotting)
    best_eps_traj: Optional[TrajectoryResult] = None
    best_dv_traj: Optional[TrajectoryResult] = None


def run_narrowed_sweep(
    encounter: TwoBodyEncounter,
    envelope: EnvelopeParams,
    num_v: int = 20,
    num_b: int = 100,
    num_angles: int = 100,
    r_start_km: float = 1.0e11,
    log_b: bool = True,
    verbose: bool = True,
) -> NarrowedBaselineResult:
    """
    Run a TwoBodyScatter sweep over the narrowed parameter envelope.

    Unlike scan_parameter_space in twobody.py (which sweeps b × angle
    at a single v_approach), this sweeps v × b × angle so that the
    baseline covers the full velocity range seen by 3-body candidates.

    Parameters
    ----------
    encounter : TwoBodyEncounter
        2-body encounter engine (star or planet).
    envelope : EnvelopeParams
        Parameter envelope from extract_envelope.
    num_v : int
        Number of v_approach samples.
    num_b : int
        Number of impact-parameter samples.
    num_angles : int
        Number of approach-angle samples.
    r_start_km : float
        Initial radial separation (km).
    log_b : bool
        Use log-spacing for impact parameter.

    Returns
    -------
    NarrowedBaselineResult
    """
    # Build grids
    v_grid = np.linspace(envelope.v_approach_min, envelope.v_approach_max, num_v)
    if log_b and envelope.b_min > 0:
        b_grid = np.logspace(
            np.log10(envelope.b_min),
            np.log10(envelope.b_max),
            num_b,
        )
    else:
        b_grid = np.linspace(envelope.b_min, envelope.b_max, num_b)
    angle_grid = np.linspace(envelope.angle_min, envelope.angle_max, num_angles)

    vstar_vec = envelope.vstar_vec if hasattr(envelope, "vstar_vec") else (0.0, envelope.vstar0)
    vstar_x, vstar_y = float(vstar_vec[0]), float(vstar_vec[1])
    total = num_v * num_b * num_angles
    if verbose:
        print(f"[{encounter.label}] Narrowed sweep: {num_v}×{num_b}×{num_angles} = {total} encounters …")

    epsilons: List[float] = []
    deltaVs: List[float] = []
    sample_v: List[float] = []
    sample_b: List[float] = []
    sample_a: List[float] = []
    sample_e: List[float] = []
    sample_dv: List[float] = []
    sample_vxf: List[float] = []
    sample_vyf: List[float] = []
    best_eps_val = -np.inf
    best_dv_val = -np.inf
    best_eps_traj: Optional[TrajectoryResult] = None
    best_dv_traj: Optional[TrajectoryResult] = None
    valid_count = 0
    count = 0

    for v_approach in v_grid:
        for angle in angle_grid:
            vx = v_approach * np.cos(angle)
            vy = v_approach * np.sin(angle)
            for b_mag in b_grid:
                perp = angle + np.pi / 2
                xm0 = -vx / v_approach * r_start_km + b_mag * np.cos(perp)
                ym0 = -vy / v_approach * r_start_km + b_mag * np.sin(perp)
                um0 = vx + vstar_x
                vm0 = vy + vstar_y

                traj = encounter.compute_trajectory(
                    xm0, ym0, um0, vm0, (vstar_x, vstar_y), num_points=50,
                )
                count += 1

                if traj.valid:
                    eps = traj.epsilon
                    dv = traj.deltaV  # vector ΔV magnitude from TwoBodyScatter
                    epsilons.append(eps)
                    deltaVs.append(dv)
                    sample_v.append(float(v_approach))
                    sample_b.append(float(b_mag))
                    sample_a.append(float(angle))
                    sample_e.append(float(traj.orbital_energy))
                    sample_dv.append(float(dv))
                    sample_vxf.append(float(traj.umF - vstar_x))
                    sample_vyf.append(float(traj.vmF - vstar_y))
                    valid_count += 1

                    if eps > best_eps_val:
                        best_eps_val = eps
                        best_eps_traj = traj
                    if dv > best_dv_val:
                        best_dv_val = dv
                        best_dv_traj = traj

                if verbose and count % max(1, total // 5) == 0:
                    print(f"  {count}/{total} ({valid_count} valid) …")

    if verbose:
        print(f"[{encounter.label}] Complete: {valid_count}/{total} successful")

    eps_arr = np.array(epsilons) if epsilons else np.array([])
    dv_arr = np.array(deltaVs) if deltaVs else np.array([])
    sample_v_arr = np.array(sample_v) if sample_v else np.array([])
    sample_b_arr = np.array(sample_b) if sample_b else np.array([])
    sample_a_arr = np.array(sample_a) if sample_a else np.array([])
    sample_e_arr = np.array(sample_e) if sample_e else np.array([])
    sample_dv_arr = np.array(sample_dv) if sample_dv else np.array([])
    sample_vxf_arr = np.array(sample_vxf) if sample_vxf else np.array([])
    sample_vyf_arr = np.array(sample_vyf) if sample_vyf else np.array([])

    return NarrowedBaselineResult(
        label=encounter.label,
        envelope=envelope,
        max_epsilon=best_eps_val if epsilons else 0.0,
        max_deltaV_vec=best_dv_val if deltaVs else 0.0,
        max_energy_half_dv_vec_sq=0.5 * best_dv_val**2 if deltaVs else 0.0,
        n_valid=valid_count,
        n_total=total,
        all_epsilons=eps_arr,
        all_deltaVs=dv_arr,
        sample_v_approach=sample_v_arr,
        sample_b=sample_b_arr,
        sample_angle=sample_a_arr,
        sample_energy=sample_e_arr,
        sample_deltaV=sample_dv_arr,
        sample_vx_final_rel=sample_vxf_arr,
        sample_vy_final_rel=sample_vyf_arr,
        best_eps_traj=best_eps_traj,
        best_dv_traj=best_dv_traj,
    )


def compute_narrowed_baselines(
    analyses_top: List[Dict[str, Any]],
    cfg,
    padding_factor: float = 1.5,
    num_v: int = 20,
    num_b: int = 100,
    num_angles: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Top-level entry point: extract envelope from top 3-body candidates,
    run velocity-matched 2-body sweeps for star and planet.

    Parameters
    ----------
    analyses_top : list of dict
        Analysis dicts from analyze_trajectory for top 3-body candidates.
    cfg : FullConfig or dict
        Configuration (used for masses and r_start).
    padding_factor : float
        How much to pad the envelope (1.5 = 50% wider each side).
    num_v / num_b / num_angles : int
        Grid resolution for the swept parameters.

    Returns
    -------
    dict
        Keys:
        - "envelope": EnvelopeParams
        - "star": NarrowedBaselineResult (or None)
        - "planet": NarrowedBaselineResult (or None)
        - "summary": str  (human-readable)
    """
    # --- resolve config ---
    if hasattr(cfg, "system"):
        M_star_Msun = cfg.system.M_star_Msun
        M_planet_Mjup = cfg.system.M_planet_Mjup
    elif isinstance(cfg, dict):
        sys_cfg = cfg.get("system", {})
        M_star_Msun = sys_cfg.get("M_star_Msun", 1.19)
        M_planet_Mjup = sys_cfg.get("M_planet_Mjup", 5.2)
    else:
        M_star_Msun = 1.19
        M_planet_Mjup = 5.2

    r_start_km = 1.0e11
    if hasattr(cfg, "two_body") and cfg.two_body is not None:
        r_start_km = cfg.two_body.r_start_km

    # --- extract envelope ---
    envelope = extract_envelope(analyses_top, padding_factor=padding_factor)
    if envelope is None:
        return {
            "envelope": None,
            "star": None,
            "planet": None,
            "summary": "No valid candidates — cannot build envelope.",
        }

    if verbose:
        print(envelope.summary())

    # --- resolve physical radii ---
    R_star_Rsun = getattr(cfg.system, 'R_star_Rsun', 1.0) if hasattr(cfg, 'system') else 1.0
    R_planet_Rjup = getattr(cfg.system, 'R_planet_Rjup', 1.155) if hasattr(cfg, 'system') else 1.155
    R_star_km = R_star_Rsun * R_SUN
    R_planet_km = R_planet_Rjup * R_JUP

    # --- run sweeps ---
    star_enc = TwoBodyEncounter(M_star_Msun * M_SUN, G_KM, label="star",
                                R_body_km=R_star_km)
    planet_enc = TwoBodyEncounter(M_planet_Mjup * M_JUP, G_KM, label="planet",
                                  R_body_km=R_planet_km)

    star_result = run_narrowed_sweep(
        star_enc, envelope,
        num_v=num_v, num_b=num_b, num_angles=num_angles,
        r_start_km=r_start_km,
        verbose=verbose,
    )
    planet_result = run_narrowed_sweep(
        planet_enc, envelope,
        num_v=num_v, num_b=num_b, num_angles=num_angles,
        r_start_km=r_start_km,
        verbose=verbose,
    )

    # --- summary ---
    lines = [
        "=" * 60,
        "  NARROWED BASELINE COMPARISON",
        "  Parameter envelope derived from top 3-body candidates",
        "=" * 60,
        envelope.summary(),
        "-" * 60,
        f"  Star  2-body: max 0.5|dV|^2 = {star_result.max_energy_half_dv_vec_sq:.4f} km^2/s^2  |  "
        f"max |dV_vec| = {star_result.max_deltaV_vec:.2f} km/s  "
        f"({star_result.n_valid}/{star_result.n_total} valid)",
        f"  Planet 2-body: max 0.5|dV|^2 = {planet_result.max_energy_half_dv_vec_sq:.4f} km^2/s^2  |  "
        f"max |dV_vec| = {planet_result.max_deltaV_vec:.2f} km/s  "
        f"({planet_result.n_valid}/{planet_result.n_total} valid)",
        f"  (epsilon = 0.5*v_inf^2 ~= {star_result.max_epsilon:.1f}; same for both, approach KE not scattering energy)",
        "",
        "  Planet-only ceiling for slingshot comparison:",
        f"    max scalar |dV_vec| = {planet_result.max_deltaV_vec:.2f} km/s,  "
        f"0.5|dV|^2 = {planet_result.max_energy_half_dv_vec_sq:.2f} km^2/s^2",
        "=" * 60,
    ]
    summary_str = "\n".join(lines)
    if verbose:
        print(summary_str)

    return {
        "envelope": envelope,
        "star": star_result,
        "planet": planet_result,
        "summary": summary_str,
    }
