"""
Baseline models and comparative analysis.
Two-body hyperbola, monopole approximation, comparison plots.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any, Tuple

from .analysis import EncounterGeometry
from .constants import G_KM as G, M_SUN, M_JUP


def two_body_hyperbola_from_state(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    mu: float,
) -> Dict[str, Any]:
    """
    Compute hyperbolic orbital elements from position and velocity.
    
    Parameters
    ----------
    r_vec : np.ndarray
        Position vector (km)
    v_vec : np.ndarray
        Velocity vector (km/s)
    mu : float
        G*M for the central body (km³/s²)
    
    Returns
    -------
    dict
        Hyperbolic parameters: a, e, v_inf, delta (deflection angle), r_p, h, etc.
        hyperbolic=False if orbit is elliptic/parabolic.
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Specific angular momentum (2D cross product)
    h_vec_3d = np.cross([r_vec[0], r_vec[1], 0.0], [v_vec[0], v_vec[1], 0.0])
    h = np.linalg.norm(h_vec_3d)
    
    # Specific energy
    eps = 0.5 * v * v - mu / r
    
    if eps <= 0:
        # Not hyperbolic
        return {
            "hyperbolic": False,
            "eps": eps,
            "a": None,
            "e": None,
            "v_inf": None,
            "deflection_deg": None,
            "r_p": None,
            "h": h,
        }
    
    # Hyperbolic elements
    a = -mu / (2.0 * eps)  # a < 0 for hyperbola
    e = np.sqrt(1.0 + 2.0 * eps * h * h / (mu * mu))
    v_inf = np.sqrt(2.0 * eps)
    
    # Turning angle (radians -> degrees)
    deflection_deg = 2.0 * np.degrees(np.arcsin(1.0 / e))
    
    # Periapsis distance
    r_p = h * h / (mu * (1.0 + e))
    
    return {
        "hyperbolic": True,
        "a": a,
        "e": e,
        "eps": eps,
        "v_inf": v_inf,
        "deflection_deg": deflection_deg,
        "r_p": r_p,
        "h": h,
    }


def sample_hyperbola_orbit(
    two_body: Dict[str, Any],
    n_pts: int = 2000,
    r_factor: float = 5.0,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Sample hyperbolic orbit near periapsis.
    
    Parameters
    ----------
    two_body : dict
        Output from two_body_hyperbola_from_state
    n_pts : int
        Number of sample points
    r_factor : float
        If r_max is None, keep r <= r_factor * r_p
    r_max : float, optional
        Absolute max radius. If given, overrides r_factor.
    
    Returns
    -------
    x_pf, y_pf, r_cut : arrays and float
        Hyperbola coordinates in periapsis frame and radius cutoff used
    """
    a = two_body["a"]
    e = two_body["e"]
    r_p = two_body["r_p"]
    
    # True anomaly at asymptote
    f_inf = np.arccos(-1.0 / e)
    f_max = 0.99 * f_inf
    f = np.linspace(-f_max, f_max, n_pts)
    
    # Hyperbolic r(f)
    r = a * (e * e - 1.0) / (1.0 + e * np.cos(f))
    r = np.abs(r)
    
    # Choose cutoff
    if r_max is None:
        if r_factor <= 1.0:
            raise ValueError(f"r_factor must be > 1, got {r_factor}")
        r_cut = r_factor * r_p
    else:
        r_cut = r_max
    
    # Keep only near-periapsis region
    mask = r <= r_cut
    if not np.any(mask):
        r_trim = r
        f_trim = f
    else:
        r_trim = r[mask]
        f_trim = f[mask]
    
    x_pf = r_trim * np.cos(f_trim)
    y_pf = r_trim * np.sin(f_trim)
    
    return x_pf, y_pf, r_cut


def hyperbola_to_planet_frame(
    two_body: Dict[str, Any],
    enc: EncounterGeometry,
    x_pf: np.ndarray,
    y_pf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate hyperbola from periapsis frame to planet frame.
    
    Parameters
    ----------
    two_body : dict
        Hyperbolic parameters
    enc : EncounterGeometry
        Extracted encounter geometry
    x_pf, y_pf : arrays
        Hyperbola in periapsis frame
    
    Returns
    -------
    x, y : arrays
        Hyperbola in planet frame
    """
    r_p_vec = enc.r_rel_p
    v_p_vec = enc.v_rel_p
    
    r_p_hat = r_p_vec / np.linalg.norm(r_p_vec)
    v_p_hat = v_p_vec / np.linalg.norm(v_p_vec)
    
    # Rotation matrix from periapsis frame to planet frame
    R = np.column_stack([r_p_hat, v_p_hat])  # 2x2
    
    pos_pf = np.vstack([x_pf, y_pf])  # 2 x N
    pos_planet = R @ pos_pf
    
    return pos_planet[0], pos_planet[1]


def monopole_ode(
    t: float,
    Y: np.ndarray,
    M_tot: float,
) -> np.ndarray:
    """
    2-body ODE with point mass M_tot at origin.
    
    Parameters
    ----------
    t : float
        Time (unused)
    Y : np.ndarray
        State [x, y, vx, vy]
    M_tot : float
        Total mass (kg)
    
    Returns
    -------
    np.ndarray
        State derivative [vx, vy, ax, ay]
    """
    x, y, vx, vy = Y
    r2 = x * x + y * y
    r = np.sqrt(r2)
    r3 = r2 * r
    ax = -G * M_tot * x / r3
    ay = -G * M_tot * y / r3
    return np.array([vx, vy, ax, ay])


def simulate_monopole_baseline(
    r_in: np.ndarray,
    v_in: np.ndarray,
    t_in: float,
    t_out: float,
    M_tot: float,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Integrate 2-body monopole baseline.
    
    Parameters
    ----------
    r_in : np.ndarray
        Initial position (km)
    v_in : np.ndarray
        Initial velocity (km/s)
    t_in, t_out : float
        Time range (s)
    M_tot : float
        Total mass (kg)
    rtol, atol : float
        Tolerances
    
    Returns
    -------
    sol, r_out, v_out
        Solution object, final position, final velocity. Returns (None, None, None) if failed.
    """
    dt = t_out - t_in
    Y0 = np.array([r_in[0], r_in[1], v_in[0], v_in[1]], dtype=float)
    
    try:
        sol = solve_ivp(
            monopole_ode,
            (0.0, dt),
            Y0,
            args=(M_tot,),
            rtol=rtol,
            atol=atol,
            dense_output=False,
            method='RK45',
        )
    except Exception as e:
        print(f"Monopole integration failed: {e}")
        return None, None, None
    
    if not sol.success:
        print(f"Monopole integration unsuccessful: {sol.message}")
        return None, None, None
    
    x_end, y_end, vx_end, vy_end = sol.y[:, -1]
    r_out = np.array([x_end, y_end])
    v_out = np.array([vx_end, vy_end])
    
    return sol, r_out, v_out


def energy_and_angmom(
    r: np.ndarray,
    v: np.ndarray,
    M_tot: float,
) -> Tuple[float, float]:
    """
    Compute specific orbital energy and angular momentum magnitude.
    
    Parameters
    ----------
    r : np.ndarray
        Position (km)
    v : np.ndarray
        Velocity (km/s)
    M_tot : float
        Total mass (kg)
    
    Returns
    -------
    eps, h : float, float
        Specific energy (km²/s²) and angular momentum magnitude (km²/s)
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    eps = 0.5 * v_mag * v_mag - G * M_tot / r_mag
    
    h_vec_3d = np.cross([r[0], r[1], 0.0], [v[0], v[1], 0.0])
    h = np.linalg.norm(h_vec_3d)
    
    return eps, h


def compare_3body_with_baselines(
    sol: object,
    enc: EncounterGeometry,
    m_star: float,
    m_p: float,
    R_p: float,
    make_plots: bool = True,
    plot_save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare 3-body trajectory with 2-body hyperbola and monopole baselines.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated 3-body trajectory
    enc : EncounterGeometry
        Extracted encounter geometry
    m_star, m_p : float
        Masses (kg)
    R_p : float
        Planet radius (km)
    make_plots : bool
        Generate comparison plots
    plot_save_dir : str, optional
        Directory to save plots. If None, plots are shown.
    
    Returns
    -------
    dict
        Comprehensive comparison metrics
    """
    if not enc.ok:
        return {"ok": False, "reason": "invalid_encounter"}
    
    M_tot = m_star + m_p
    mu_p = G * m_p
    
    # 2-body hyperbola
    two_body = two_body_hyperbola_from_state(enc.r_rel_i, enc.v_rel_i, mu_p)
    
    # Monopole baseline
    sol_mono, r_out_mono, v_out_mono = simulate_monopole_baseline(
        enc.r_in_bary, enc.v_in_bary,
        enc.t_in, enc.t_out,
        M_tot,
    )
    
    if sol_mono is None:
        return {"ok": False, "reason": "monopole_integration_failed"}
    
    # Energy and angular momentum
    eps_in, h_in = energy_and_angmom(enc.r_in_bary, enc.v_in_bary, M_tot)
    eps_3b, h_3b = energy_and_angmom(enc.r_out_bary, enc.v_out_bary, M_tot)
    eps_0, h_0 = energy_and_angmom(r_out_mono, v_out_mono, M_tot)
    
    # Calculate deflection angle for 3-body encounter
    # Extract velocity components from velocity vectors (handle 2D and 3D)
    v_in_2d = enc.v_in_bary[:2] if len(enc.v_in_bary) >= 2 else enc.v_in_bary
    v_out_2d = enc.v_out_bary[:2] if len(enc.v_out_bary) >= 2 else enc.v_out_bary
    
    vx_i, vy_i = float(v_in_2d[0]), float(v_in_2d[1])
    vx_f, vy_f = float(v_out_2d[0]), float(v_out_2d[1])
    
    v_inf_in_mag = np.hypot(vx_i, vy_i)
    v_inf_out_mag = np.hypot(vx_f, vy_f)
    
    # Calculate deflection angle safely
    if v_inf_in_mag > 1e-6 and v_inf_out_mag > 1e-6:
        # Dot product between incoming and outgoing velocities
        dot_product = vx_i * vx_f + vy_i * vy_f
        
        # Cosine of deflection angle (clipped to [-1, 1] to avoid numerical issues)
        cos_deflection = dot_product / (v_inf_in_mag * v_inf_out_mag)
        cos_deflection = np.clip(float(cos_deflection), -1.0, 1.0)
        
        # Deflection angle in degrees
        deflection_angle_3body = np.degrees(np.arccos(cos_deflection))
    else:
        deflection_angle_3body = np.nan
    
    result = {
        "ok": True,
        "two_body": two_body,
        "encounter": enc,
        "bary_eps_in": eps_in,
        "bary_eps_3b": eps_3b,
        "bary_eps_0": eps_0,
        "bary_h_in": h_in,
        "bary_h_3b": h_3b,
        "bary_h_0": h_0,
        "delta_eps_3b": eps_3b - eps_in,
        "delta_eps_0": eps_0 - eps_in,
        "extra_eps_from_planet": eps_3b - eps_0,
        "delta_h_3b": h_3b - h_in,
        "delta_h_0": h_0 - h_in,
        "extra_h_from_planet": h_3b - h_0,
        "deflection_3body": deflection_angle_3body,
    }
    
    if make_plots:
        import matplotlib.pyplot as plt
        
        # Planet-frame comparison
        if two_body["hyperbolic"]:
            x_pf, y_pf, r_cut = sample_hyperbola_orbit(two_body, n_pts=1000, r_factor=5.0)
            x_hyp, y_hyp = hyperbola_to_planet_frame(two_body, enc, x_pf, y_pf)
            
            # 3-body planet frame
            y_all = sol.y
            xp, yp = y_all[4], y_all[5]
            xsat, ysat = y_all[8], y_all[9]
            dx = xsat - xp
            dy = ysat - yp
            r_rel = np.hypot(dx, dy)
            
            mask3 = r_rel <= r_cut
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_hyp, y_hyp, label="2-body hyperbola", linewidth=2, alpha=0.7)
            ax.plot(dx[mask3], dy[mask3], label="3-body (planet frame)", linewidth=2, alpha=0.7)
            ax.scatter(0, 0, s=100, c='k', marker='*', label='Planet', zorder=5)
            ax.set_aspect('equal')
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_title('Planet-frame comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if plot_save_dir:
                fig.savefig(f"{plot_save_dir}/planet_frame_comparison.png", dpi=150)
            plt.show()
        
        # Barycentric comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        xsat_3b = sol.y[8]
        ysat_3b = sol.y[9]
        x_star = sol.y[0]
        y_star = sol.y[1]
        xsat_0 = sol_mono.y[0]
        ysat_0 = sol_mono.y[1]
        
        ax.plot(x_star, y_star, label="Star trajectory", linewidth=1, alpha=0.6)
        ax.plot(xsat_3b, ysat_3b, label="3-body trajectory", linewidth=2, alpha=0.8)
        ax.plot(xsat_0, ysat_0, '--', label="Monopole baseline", linewidth=2, alpha=0.8)
        ax.scatter(0, 0, s=100, c='k', marker='+', label='Barycenter', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('Barycentric comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if plot_save_dir:
            fig.savefig(f"{plot_save_dir}/barycentric_comparison.png", dpi=150)
        plt.show()
    
    return result
