"""
Baseline models and comparative analysis.
Two-body hyperbola, monopole approximation, comparison plots.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any, Tuple

from .trajectory import EncounterGeometry
from ..constants import G_KM as G, M_SUN, M_JUP


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
    r_center0: np.ndarray,
    v_center: np.ndarray,
) -> np.ndarray:
    """
    2-body ODE with point mass M_tot at a center that can drift linearly.
    
    Parameters
    ----------
    t : float
        Time (unused)
    Y : np.ndarray
        State [x, y, vx, vy]
    M_tot : float
        Total mass (kg)
    r_center0 : np.ndarray
        Center position at t=0 (km), shape (2,).
    v_center : np.ndarray
        Constant center velocity (km/s), shape (2,).
    
    Returns
    -------
    np.ndarray
        State derivative [vx, vy, ax, ay]
    """
    x, y, vx, vy = Y
    xc = r_center0[0] + v_center[0] * t
    yc = r_center0[1] + v_center[1] * t
    dx = x - xc
    dy = y - yc
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    r3 = r2 * r
    ax = -G * M_tot * dx / r3
    ay = -G * M_tot * dy / r3
    return np.array([vx, vy, ax, ay])


def simulate_monopole_baseline(
    r_in: np.ndarray,
    v_in: np.ndarray,
    t_in: float,
    t_out: float,
    M_tot: float,
    r_center0: Optional[np.ndarray] = None,
    v_center: Optional[np.ndarray] = None,
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
    r_center0 : np.ndarray, optional
        Monopole-center position at t=0 (km). Default [0, 0].
    v_center : np.ndarray, optional
        Constant monopole-center velocity (km/s). Default [0, 0].
    rtol, atol : float
        Tolerances
    
    Returns
    -------
    sol, r_out, v_out
        Solution object, final position, final velocity. Returns (None, None, None) if failed.
    """
    dt = t_out - t_in
    if r_center0 is None:
        r_center0 = np.array([0.0, 0.0], dtype=float)
    if v_center is None:
        v_center = np.array([0.0, 0.0], dtype=float)
    r_center0 = np.asarray(r_center0, dtype=float)
    v_center = np.asarray(v_center, dtype=float)
    Y0 = np.array([r_in[0], r_in[1], v_in[0], v_in[1]], dtype=float)
    
    try:
        sol = solve_ivp(
            monopole_ode,
            (0.0, dt),
            Y0,
            args=(M_tot, r_center0, v_center),
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
    if enc.i0 is None or enc.i1 is None:
        return {"ok": False, "reason": "missing_encounter_indices"}
    
    M_tot = m_star + m_p
    mu_p = G * m_p
    y_all = sol.y
    i0, i1 = int(enc.i0), int(enc.i1)

    # COM state at encounter entry/exit (includes optional bulk drift).
    xs_i, ys_i = y_all[0, i0], y_all[1, i0]
    vxs_i, vys_i = y_all[2, i0], y_all[3, i0]
    xp_i, yp_i = y_all[4, i0], y_all[5, i0]
    vxp_i, vyp_i = y_all[6, i0], y_all[7, i0]
    xs_f, ys_f = y_all[0, i1], y_all[1, i1]
    xp_f, yp_f = y_all[4, i1], y_all[5, i1]
    vxs_f, vys_f = y_all[2, i1], y_all[3, i1]
    vxp_f, vyp_f = y_all[6, i1], y_all[7, i1]

    r_com_in = np.array([
        (m_star * xs_i + m_p * xp_i) / M_tot,
        (m_star * ys_i + m_p * yp_i) / M_tot,
    ])
    v_com_in = np.array([
        (m_star * vxs_i + m_p * vxp_i) / M_tot,
        (m_star * vys_i + m_p * vyp_i) / M_tot,
    ])
    r_com_out = np.array([
        (m_star * xs_f + m_p * xp_f) / M_tot,
        (m_star * ys_f + m_p * yp_f) / M_tot,
    ])
    v_com_out = np.array([
        (m_star * vxs_f + m_p * vxp_f) / M_tot,
        (m_star * vys_f + m_p * vyp_f) / M_tot,
    ])
    
    # 2-body hyperbola
    two_body = two_body_hyperbola_from_state(enc.r_rel_i, enc.v_rel_i, mu_p)
    
    # Monopole baseline (moving COM center, not fixed origin)
    sol_mono, r_out_mono, v_out_mono = simulate_monopole_baseline(
        enc.r_in_bary, enc.v_in_bary,
        enc.t_in, enc.t_out,
        M_tot,
        r_center0=r_com_in,
        v_center=v_com_in,
    )
    
    if sol_mono is None:
        return {"ok": False, "reason": "monopole_integration_failed"}
    
    # Energy and angular momentum in COM-relative coordinates.
    dt = float(enc.t_out - enc.t_in)
    r_com_out_lin = r_com_in + v_com_in * dt
    r_in_rel = enc.r_in_bary - r_com_in
    v_in_rel = enc.v_in_bary - v_com_in
    r_out_3b_rel = enc.r_out_bary - r_com_out
    v_out_3b_rel = enc.v_out_bary - v_com_out
    r_out_0_rel = r_out_mono - r_com_out_lin
    v_out_0_rel = v_out_mono - v_com_in

    eps_in, h_in = energy_and_angmom(r_in_rel, v_in_rel, M_tot)
    eps_3b, h_3b = energy_and_angmom(r_out_3b_rel, v_out_3b_rel, M_tot)
    eps_0, h_0 = energy_and_angmom(r_out_0_rel, v_out_0_rel, M_tot)
    
    # Deflection angle in COM frame for boost-invariant parity.
    v_in_2d = v_in_rel[:2] if len(v_in_rel) >= 2 else v_in_rel
    v_out_2d = v_out_3b_rel[:2] if len(v_out_3b_rel) >= 2 else v_out_3b_rel
    
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
        "com_r_in": r_com_in,
        "com_v_in": v_com_in,
        "com_r_out": r_com_out,
        "com_v_out": v_com_out,
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
                plt.close(fig)
            else:
                plt.show()
        
        # Barycentric comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        xsat_3b = sol.y[8]
        ysat_3b = sol.y[9]
        x_star = sol.y[0]
        y_star = sol.y[1]
        t_mono = sol_mono.t
        x_com_path = r_com_in[0] + v_com_in[0] * t_mono
        y_com_path = r_com_in[1] + v_com_in[1] * t_mono
        xsat_0 = sol_mono.y[0] + x_com_path
        ysat_0 = sol_mono.y[1] + y_com_path
        
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
            plt.close(fig)
        else:
            plt.show()
    
    return result
