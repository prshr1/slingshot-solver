"""
Core dynamics for restricted 3-body problem: star + hot Jupiter + massless satellite.
All in barycentric (inertial) frame.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple

from .constants import G_KM as G, M_SUN, M_JUP, R_JUP, AU_KM


def init_hot_jupiter_barycentric(
    a_km: Optional[float] = None,
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    phase: float = 0.0,
    prograde: bool = True,
    bulk_velocity_vx_kms: float = 0.0,
    bulk_velocity_vy_kms: float = 0.0,
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
    bulk_velocity_vx_kms : float
        Uniform system bulk velocity x-component (km/s) added to both
        star and planet.
    bulk_velocity_vy_kms : float
        Uniform system bulk velocity y-component (km/s) added to both
        star and planet.
    
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

    # Optional Galilean boost for moving star systems
    vxs += bulk_velocity_vx_kms
    vys += bulk_velocity_vy_kms
    vxp += bulk_velocity_vx_kms
    vyp += bulk_velocity_vy_kms
    
    return np.array([xs, ys, vxs, vys, xp, yp, vxp, vyp], dtype=float)


def restricted_3body_ode(
    t: float,
    Y: np.ndarray,
    m_star: float,
    m_p: float,
    eps2: float = 0.0,
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
    eps2 : float
        Plummer softening length squared (km²).
        Replaces r³ with (r² + ε²)^{3/2} for satellite-body forces only.
        The star-planet mutual force is never softened.
        Set to 0 for pure Newtonian gravity.
    
    Returns
    -------
    np.ndarray
        Time derivative dY/dt
    """
    xs, ys, vxs, vys, xp, yp, vxp, vyp, x3, y3, vx3, vy3 = Y
    
    # Star-planet separation  (NOT softened — stable circular orbit)
    dx_sp = xp - xs
    dy_sp = yp - ys
    r_sp2 = dx_sp * dx_sp + dy_sp * dy_sp
    r_sp = np.sqrt(r_sp2)
    r_sp3 = r_sp2 * r_sp if r_sp > 0 else np.inf
    
    # Satellite-star separation (vector from sat -> star)
    dx_3s = xs - x3
    dy_3s = ys - y3
    r_3s2 = dx_3s * dx_3s + dy_3s * dy_3s
    r_3s3 = (r_3s2 + eps2) ** 1.5       # Plummer-softened denominator
    
    # Satellite-planet separation (vector from sat -> planet)
    dx_3p = xp - x3
    dy_3p = yp - y3
    r_3p2 = dx_3p * dx_3p + dy_3p * dy_3p
    r_3p3 = (r_3p2 + eps2) ** 1.5       # Plummer-softened denominator
    
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
    escape_radius_km: Optional[float] = None,
    method: str = 'DOP853',
    softening_km: float = 0.0,
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
    escape_radius_km : float, optional
        If set, integration terminates when the satellite recedes past this
        radius from the barycenter AND is moving outward (after periapsis).
        This avoids wasting time on particles that have already completed
        their flyby.  If None, no early termination.
    method : str
        ODE solver method for scipy.integrate.solve_ivp.
        'DOP853' (default) is order-8 explicit, best for orbital mechanics.
        'Radau' is implicit, handles stiff close encounters better.
        'RK45' is the scipy default but degrades badly during close passes.
    softening_km : float
        Plummer softening length (km) for satellite–body forces.
        Replaces r³ with (r² + ε²)^{3/2}, capping force gradients at close
        approach and preventing adaptive-step collapse.  Not applied to the
        star–planet mutual force.  Set to 0 for pure Newtonian gravity.
    
    Returns
    -------
    scipy.integrate.OdeResult or None
        Solution object if successful, None otherwise
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    
    # Softening: convert length to squared (ODE expects eps²)
    eps2 = softening_km * softening_km
    
    if n_eval is not None and n_eval > 0:
        t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    else:
        t_eval = None   # adaptive solver steps — dense near close encounters
    
    # Build terminal event list
    events = None
    if escape_radius_km is not None:
        r_esc = float(escape_radius_km)

        def _escape_event(t, Y, m_star, m_p, eps2):
            """Fires when satellite distance from barycenter crosses r_esc outward."""
            x3, y3 = Y[8], Y[9]
            return np.sqrt(x3 * x3 + y3 * y3) - r_esc

        _escape_event.terminal = True
        _escape_event.direction = +1.0   # only outward crossing
        events = [_escape_event]
    
    try:
        sol = solve_ivp(
            restricted_3body_ode,
            t_span,
            Y0,
            args=(m_star, m_p, eps2),
            rtol=rtol,
            atol=atol,
            dense_output=False,
            t_eval=t_eval,
            events=events,
            method=method,
        )
    except Exception as e:
        print(f"Integration failed: {e}")
        return None
    
    if not sol.success:
        print(f"Integration unsuccessful: {sol.message}")
        return None
    
    return sol
