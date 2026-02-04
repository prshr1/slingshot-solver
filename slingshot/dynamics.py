"""
Core dynamics for restricted 3-body problem: star + hot Jupiter + massless satellite.
All in barycentric (inertial) frame.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple


# Physical constants
G = 6.67430e-20  # km^3 / (kg s^2)
M_SUN = 1.98847e30  # kg
M_JUP = 1.898e27  # kg
R_JUP = 71492.0  # km
AU_KM = 1.495978707e8  # km


def init_hot_jupiter_barycentric(
    a_km: Optional[float] = None,
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    phase: float = 0.0,
    prograde: bool = True,
) -> np.ndarray:
    """
    Generate barycentric initial conditions for star+planet 2-body system
    on circular orbit.
    
    Parameters
    ----------
    a_km : float, optional
        Orbital semi-major axis (km). If None, uses default (0.0896 AU).
    m_star : float, optional
        Star mass (kg). If None, uses 1 solar mass.
    m_p : float, optional
        Planet mass (kg). If None, uses 1 Jupiter mass.
    phase : float
        Orbital phase (radians). Default 0.0 (planet at +x).
    prograde : bool
        If True, orbit is counter-clockwise. If False, clockwise.
    
    Returns
    -------
    np.ndarray
        State vector [xs, ys, vxs, vys, xp, yp, vxp, vyp] (barycentric)
    """
    if a_km is None:
        a_km = 0.0896 * AU_KM
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    
    M_tot = m_star + m_p
    mu_tot = G * M_tot
    
    r = a_km
    v_rel = np.sqrt(mu_tot / r)  # relative orbital speed
    
    # Relative position (planet wrt star)
    x_rel = r * np.cos(phase)
    y_rel = r * np.sin(phase)
    
    # Relative velocity (planet wrt star), 90° ahead for circular orbit
    sign = 1.0 if prograde else -1.0
    vx_rel = -sign * v_rel * np.sin(phase)
    vy_rel = sign * v_rel * np.cos(phase)
    
    # Barycentric factors
    f_star = m_p / M_tot  # |r_star| = f_star * r
    f_plan = m_star / M_tot
    
    xs = -f_star * x_rel
    ys = -f_star * y_rel
    xp = f_plan * x_rel
    yp = f_plan * y_rel
    
    vxs = -f_star * vx_rel
    vys = -f_star * vy_rel
    vxp = f_plan * vx_rel
    vyp = f_plan * vy_rel
    
    return np.array([xs, ys, vxs, vys, xp, yp, vxp, vyp], dtype=float)


def restricted_3body_ode(
    t: float,
    Y: np.ndarray,
    m_star: float,
    m_p: float,
) -> np.ndarray:
    """
    Restricted planar 3-body ODE in barycentric frame.
    
    System:
      - Body 0: Star (mass m_star)
      - Body 1: Planet (mass m_p)
      - Body 2: Satellite (massless, feels both but doesn't back-react)
    
    Parameters
    ----------
    t : float
        Time (not used directly, needed for scipy.integrate.solve_ivp signature)
    Y : np.ndarray
        State vector [xs, ys, vxs, vys, xp, yp, vxp, vyp, x3, y3, vx3, vy3]
    m_star : float
        Star mass (kg)
    m_p : float
        Planet mass (kg)
    
    Returns
    -------
    np.ndarray
        Time derivative dY/dt
    """
    xs, ys, vxs, vys, xp, yp, vxp, vyp, x3, y3, vx3, vy3 = Y
    
    # Star-planet separation
    dx_sp = xp - xs
    dy_sp = yp - ys
    r_sp2 = dx_sp * dx_sp + dy_sp * dy_sp
    r_sp = np.sqrt(r_sp2)
    r_sp3 = r_sp2 * r_sp if r_sp > 0 else np.inf
    
    # Satellite-star separation (vector from sat -> star)
    dx_3s = xs - x3
    dy_3s = ys - y3
    r_3s2 = dx_3s * dx_3s + dy_3s * dy_3s
    r_3s = np.sqrt(r_3s2)
    r_3s3 = r_3s2 * r_3s if r_3s > 0 else np.inf
    
    # Satellite-planet separation (vector from sat -> planet)
    dx_3p = xp - x3
    dy_3p = yp - y3
    r_3p2 = dx_3p * dx_3p + dy_3p * dy_3p
    r_3p = np.sqrt(r_3p2)
    r_3p3 = r_3p2 * r_3p if r_3p > 0 else np.inf
    
    # Accelerations of star and planet (due to each other)
    ax_s = G * m_p * dx_sp / r_sp3
    ay_s = G * m_p * dy_sp / r_sp3
    ax_p = -G * m_star * dx_sp / r_sp3
    ay_p = -G * m_star * dy_sp / r_sp3
    
    # Satellite acceleration from star and planet
    ax_3 = G * (m_star * dx_3s / r_3s3 + m_p * dx_3p / r_3p3)
    ay_3 = G * (m_star * dy_3s / r_3s3 + m_p * dy_3p / r_3p3)
    
    return np.array([
        vxs, vys, ax_s, ay_s,
        vxp, vyp, ax_p, ay_p,
        vx3, vy3, ax_3, ay_3,
    ], dtype=float)


def simulate_3body(
    Y0: np.ndarray,
    t_span: Tuple[float, float],
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    n_eval: Optional[int] = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> Optional[object]:
    """
    Integrate restricted 3-body system.
    
    Parameters
    ----------
    Y0 : np.ndarray
        Initial state [xs, ys, vxs, vys, xp, yp, vxp, vyp, x3, y3, vx3, vy3]
    t_span : tuple
        (t_start, t_end) integration time span
    m_star : float, optional
        Star mass. If None, uses 1 solar mass.
    m_p : float, optional
        Planet mass. If None, uses 1 Jupiter mass.
    n_eval : int, optional
        Number of output points. If None, uses solver's adaptive stepping.
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver
    
    Returns
    -------
    scipy.integrate.OdeResult or None
        Solution object if successful, None otherwise
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    
    if n_eval is not None:
        t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    else:
        t_eval = None
    
    try:
        sol = solve_ivp(
            restricted_3body_ode,
            t_span,
            Y0,
            args=(m_star, m_p),
            rtol=rtol,
            atol=atol,
            dense_output=False,
            t_eval=t_eval,
            method='RK45',
        )
    except Exception as e:
        print(f"Integration failed: {e}")
        return None
    
    if not sol.success:
        print(f"Integration unsuccessful: {sol.message}")
        return None
    
    return sol
