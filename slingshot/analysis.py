"""
Trajectory analysis for slingshot encounters.
Supports both planet-frame and barycentric-frame analysis.
Robust encounter geometry extraction.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


G = 6.67430e-20  # km^3 / (kg s^2)
M_SUN = 1.98847e30  # kg
M_JUP = 1.898e27  # kg
R_JUP = 71492.0  # km


@dataclass
class EncounterGeometry:
    """
    Encapsulation of encounter geometry and state extraction.
    Extracted from integrated 3-body trajectory.
    """
    ok: bool  # Whether geometry extraction succeeded
    reason: Optional[str] = None  # Reason for failure if not ok
    
    # Indices
    i0: Optional[int] = None  # Index of "far in" state
    i1: Optional[int] = None  # Index of "far out" state
    k_min: Optional[int] = None  # Index of periapsis (closest approach)
    
    # Planet-frame states
    r_rel_i: Optional[np.ndarray] = None  # Position relative to planet, incoming (km)
    v_rel_i: Optional[np.ndarray] = None  # Velocity relative to planet, incoming (km/s)
    r_rel_f: Optional[np.ndarray] = None  # Position relative to planet, outgoing (km)
    v_rel_f: Optional[np.ndarray] = None  # Velocity relative to planet, outgoing (km/s)
    r_rel_p: Optional[np.ndarray] = None  # Position at periapsis (km)
    v_rel_p: Optional[np.ndarray] = None  # Velocity at periapsis (km/s)
    
    # Barycentric states
    r_in_bary: Optional[np.ndarray] = None  # Position barycentric, incoming (km)
    v_in_bary: Optional[np.ndarray] = None  # Velocity barycentric, incoming (km/s)
    r_out_bary: Optional[np.ndarray] = None  # Position barycentric, outgoing (km)
    v_out_bary: Optional[np.ndarray] = None  # Velocity barycentric, outgoing (km/s)
    
    # Distances and times
    r_min: Optional[float] = None  # Minimum distance to planet (km)
    t_in: Optional[float] = None  # Time at incoming state (s)
    t_out: Optional[float] = None  # Time at outgoing state (s)


def wrap_angle_deg(angle_deg: float) -> float:
    """
    Wrap angle to [-180, 180] degrees range.
    
    Parameters
    ----------
    angle_deg : float
        Angle in degrees
    
    Returns
    -------
    float
        Wrapped angle in [-180, 180]
    """
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def extract_encounter_states(
    sol: object,
    m_p: float,
    R_p: float,
    r_far_factor: float = 20.0,
    min_clearance_factor: float = 1.05,
) -> EncounterGeometry:
    """
    Extract encounter geometry from integrated 3-body trajectory.
    
    Uses "far" distance threshold to identify asymptotic regimes.
    Returns encounter states in both planet-frame and barycentric frame.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory solution
    m_p : float
        Planet mass (kg)
    R_p : float
        Planet radius (km)
    r_far_factor : float
        Distance factor: "far" region is r > r_far_factor * R_p
    min_clearance_factor : float
        Reject if closest approach < min_clearance_factor * R_p
    
    Returns
    -------
    EncounterGeometry
        Extracted encounter states and metadata
    """
    if sol is None or sol.y.shape[1] < 3:
        return EncounterGeometry(ok=False, reason="insufficient_solution_points")
    
    y = sol.y
    t = sol.t
    
    # Unpack state indices
    xp, yp = y[4], y[5]
    vxp, vyp = y[6], y[7]
    xsat, ysat = y[8], y[9]
    vxsat, vysat = y[10], y[11]
    
    # Planet-frame position of satellite
    dx = xsat - xp
    dy = ysat - yp
    r = np.hypot(dx, dy)
    
    # Periapsis: minimum distance
    k_min = int(np.argmin(r))
    r_min = float(r[k_min])
    
    # Reject if collision/graze
    if r_min <= R_p * min_clearance_factor:
        return EncounterGeometry(
            ok=False,
            reason="collision_or_graze",
            r_min=r_min,
            k_min=k_min,
        )
    
    # Find "far" indices using r_far threshold
    r_far = R_p * r_far_factor
    far_idx = np.where(r > r_far)[0]
    
    if far_idx.size < 2:
        return EncounterGeometry(
            ok=False,
            reason="never_asymptotic",
            r_min=r_min,
            k_min=k_min,
        )
    
    i0 = int(far_idx[0])
    i1 = int(far_idx[-1])
    
    # Planet-frame velocities
    dvx = vxsat - vxp
    dvy = vysat - vyp
    
    # Extract states
    r_rel_i = np.array([dx[i0], dy[i0]])
    v_rel_i = np.array([dvx[i0], dvy[i0]])
    r_rel_f = np.array([dx[i1], dy[i1]])
    v_rel_f = np.array([dvx[i1], dvy[i1]])
    r_rel_p = np.array([dx[k_min], dy[k_min]])
    v_rel_p = np.array([dvx[k_min], dvy[k_min]])
    
    # Barycentric states
    r_in_bary = np.array([xsat[i0], ysat[i0]])
    v_in_bary = np.array([vxsat[i0], vysat[i0]])
    r_out_bary = np.array([xsat[i1], ysat[i1]])
    v_out_bary = np.array([vxsat[i1], vysat[i1]])
    
    return EncounterGeometry(
        ok=True,
        reason=None,
        i0=i0,
        i1=i1,
        k_min=k_min,
        r_rel_i=r_rel_i,
        v_rel_i=v_rel_i,
        r_rel_f=r_rel_f,
        v_rel_f=v_rel_f,
        r_rel_p=r_rel_p,
        v_rel_p=v_rel_p,
        r_in_bary=r_in_bary,
        v_in_bary=v_in_bary,
        r_out_bary=r_out_bary,
        v_out_bary=v_out_bary,
        r_min=r_min,
        t_in=float(t[i0]),
        t_out=float(t[i1]),
    )


def analyze_trajectory(
    sol: object,
    frame: str = "planet",
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    R_p: Optional[float] = None,
    r_far_factor: float = 20.0,
    min_clearance_factor: float = 1.05,
) -> Optional[Dict[str, Any]]:
    """
    Unified trajectory analysis in planet or barycentric frame.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    frame : str
        "planet" (relative to planet) or "barycentric" (inertial frame)
    m_star : float, optional
        Star mass (kg)
    m_p : float, optional
        Planet mass (kg)
    R_p : float, optional
        Planet radius (km)
    r_far_factor : float
        Factor for "far" region distance threshold
    min_clearance_factor : float
        Minimum clearance factor for collision detection
    
    Returns
    -------
    dict or None
        Analysis dict with deflection, Δv, energy metrics, etc.
        Returns None if analysis fails.
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    if R_p is None:
        R_p = R_JUP
    
    # Extract encounter geometry
    enc = extract_encounter_states(
        sol,
        m_p=m_p,
        R_p=R_p,
        r_far_factor=r_far_factor,
        min_clearance_factor=min_clearance_factor,
    )
    
    if not enc.ok:
        return None
    
    if frame.lower() == "planet":
        # Planet-frame analysis
        v_i = np.linalg.norm(enc.v_rel_i)
        v_f = np.linalg.norm(enc.v_rel_f)
        
        if v_i == 0.0 or v_f == 0.0:
            return None
        
        # Deflection angle
        theta_i = np.degrees(np.arctan2(enc.v_rel_i[1], enc.v_rel_i[0]))
        theta_f = np.degrees(np.arctan2(enc.v_rel_f[1], enc.v_rel_f[0]))
        deflection = wrap_angle_deg(theta_f - theta_i)
        
        # Angular momentum and impact parameter
        L_mag = np.abs(np.cross(enc.r_rel_i, enc.v_rel_i))
        impact_parameter = L_mag / v_i if v_i > 0 else 0.0
        
        # Specific energy wrt planet
        mu_p = G * m_p
        r_i = np.linalg.norm(enc.r_rel_i)
        r_f = np.linalg.norm(enc.r_rel_f)
        
        eps_i = 0.5 * v_i**2 - mu_p / r_i
        eps_f = 0.5 * v_f**2 - mu_p / r_f
        unbound_f = eps_f > 0.0
        
        return {
            "frame": "planet",
            "v_i": v_i,
            "v_f": v_f,
            "delta_v": v_f - v_i,
            "delta_v_pct": 100.0 * (v_f - v_i) / v_i,
            "deflection": deflection,
            "deflection_frac": deflection / 180.0,
            "r_min": enc.r_min,
            "impact_parameter": impact_parameter,
            "eps_i": eps_i,
            "eps_f": eps_f,
            "unbound_f": unbound_f,
            "encounter": enc,
        }
    
    elif frame.lower() == "barycentric":
        # Barycentric frame analysis
        v_i = np.linalg.norm(enc.v_in_bary)
        v_f = np.linalg.norm(enc.v_out_bary)
        
        if v_i == 0.0 or v_f == 0.0:
            return None
        
        # Deflection angle
        theta_i = np.degrees(np.arctan2(enc.v_in_bary[1], enc.v_in_bary[0]))
        theta_f = np.degrees(np.arctan2(enc.v_out_bary[1], enc.v_out_bary[0]))
        deflection = wrap_angle_deg(theta_f - theta_i)
        
        # Angular momentum wrt barycenter
        L_mag = np.abs(np.cross(enc.r_in_bary, enc.v_in_bary))
        impact_parameter = L_mag / v_i if v_i > 0 else 0.0
        
        # Specific energy wrt barycenter (both star and planet potentials)
        M_tot = m_star + m_p
        r_i = np.linalg.norm(enc.r_in_bary)
        r_f = np.linalg.norm(enc.r_out_bary)
        
        # Compute distances to star and planet
        y_sol = sol.y
        xs_i, ys_i = y_sol[0, enc.i0], y_sol[1, enc.i0]
        xp_i, yp_i = y_sol[4, enc.i0], y_sol[5, enc.i0]
        xs_f, ys_f = y_sol[0, enc.i1], y_sol[1, enc.i1]
        xp_f, yp_f = y_sol[4, enc.i1], y_sol[5, enc.i1]
        
        r_3s_i = np.hypot(xs_i - enc.r_in_bary[0], ys_i - enc.r_in_bary[1])
        r_3p_i = np.hypot(xp_i - enc.r_in_bary[0], yp_i - enc.r_in_bary[1])
        r_3s_f = np.hypot(xs_f - enc.r_out_bary[0], ys_f - enc.r_out_bary[1])
        r_3p_f = np.hypot(xp_f - enc.r_out_bary[0], yp_f - enc.r_out_bary[1])
        
        pot_i = 0.0
        pot_f = 0.0
        if r_3s_i > 0:
            pot_i += -G * m_star / r_3s_i
        if r_3p_i > 0:
            pot_i += -G * m_p / r_3p_i
        if r_3s_f > 0:
            pot_f += -G * m_star / r_3s_f
        if r_3p_f > 0:
            pot_f += -G * m_p / r_3p_f
        
        eps_i = 0.5 * v_i**2 + pot_i
        eps_f = 0.5 * v_f**2 + pot_f
        unbound_f = eps_f > 0.0
        unbound_i = eps_i > 0.0
        
        return {
            "frame": "barycentric",
            "v_i": v_i,
            "v_f": v_f,
            "delta_v": v_f - v_i,
            "delta_v_pct": 100.0 * (v_f - v_i) / v_i,
            "deflection": deflection,
            "deflection_frac": deflection / 180.0,
            "r_min": enc.r_min,
            "impact_parameter": impact_parameter,
            "eps_i": eps_i,
            "eps_f": eps_f,
            "unbound_i": unbound_i,
            "unbound_f": unbound_f,
            "encounter": enc,
        }
    
    else:
        raise ValueError(f"Unknown frame: {frame}. Use 'planet' or 'barycentric'.")
