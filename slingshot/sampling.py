"""
Initial condition sampling for satellites.
Supports barycentric parametric (hyperbolic) and planet-relative sampling modes.
"""

import numpy as np
from typing import Optional, Tuple

from .constants import AU_KM, R_JUP


def sample_satellite_state_barycentric(
    Y_sp0: np.ndarray,
    N: int = 1,
    v_mag_min: float = 10.0,
    v_mag_max: float = 120.0,
    impact_param_min_AU: float = 0.5,
    impact_param_max_AU: float = 3.0,
    angle_in_min_deg: float = -60.0,
    angle_in_max_deg: float = 60.0,
    r_init_AU: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample N satellite states in barycentric frame using parametric hyperbolic encounter model.
    
    Controlled parameters:
      - Velocity magnitude: asymptotic speed
      - Impact parameter: perpendicular distance to barycenter
      - Incoming angle: radial vs tangential velocity mix
      - Azimuth: orientation of encounter in plane
    
    Parameters
    ----------
    Y_sp0 : np.ndarray
        Star+planet state [xs, ys, vxs, vys, xp, yp, vxp, vyp]
    N : int
        Number of samples
    v_mag_min : float
        Min velocity magnitude (km/s)
    v_mag_max : float
        Max velocity magnitude (km/s)
    impact_param_min_AU : float
        Min impact parameter (AU)
    impact_param_max_AU : float
        Max impact parameter (AU)
    angle_in_min_deg : float
        Min incoming angle (degrees from radial)
    angle_in_max_deg : float
        Max incoming angle (degrees)
    r_init_AU : float, optional
        Fixed initial distance from barycenter (AU). If None, defaults to
        2× the sampled impact parameter.
    rng : np.random.Generator, optional
        Random number generator. If None, creates new one.
    
    Returns
    -------
    np.ndarray
        Shape (N, 4): [x3, y3, vx3, vy3] in barycentric frame
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert impact parameter to km
    impact_param_min_km = impact_param_min_AU * AU_KM
    impact_param_max_km = impact_param_max_AU * AU_KM
    
    samples = np.zeros((N, 4))
    
    for i in range(N):
        # Sample independent variables
        v_mag = rng.uniform(v_mag_min, v_mag_max)
        impact_param = rng.uniform(impact_param_min_km, impact_param_max_km)
        angle_in_deg = rng.uniform(angle_in_min_deg, angle_in_max_deg)
        angle_in_rad = np.radians(angle_in_deg)
        
        # Azimuthal orientation
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        
        # Asymptotic position and velocity in hyperbolic encounter frame
        # r_init must be >> a_planet (~0.09 AU) but need not be 10× b.
        # Use explicit r_init_AU if provided, otherwise 2× impact_param.
        if r_init_AU is not None:
            r_init = r_init_AU * AU_KM
        else:
            r_init = 2.0 * impact_param
        
        # Local coordinates: x = radial inward, y = impact parameter direction
        x_local = -r_init  # approaching from negative side
        y_local = impact_param  # perpendicular offset
        
        # Velocity in local frame: magnitude v_mag, partly radial/tangential
        vx_local = v_mag * np.cos(angle_in_rad)  # radial component
        vy_local = v_mag * np.sin(angle_in_rad)  # tangential component
        
        # Rotate from local to barycentric frame using azimuth
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        
        x3 = x_local * cos_az - y_local * sin_az
        y3 = x_local * sin_az + y_local * cos_az
        vx3 = vx_local * cos_az - vy_local * sin_az
        vy3 = vx_local * sin_az + vy_local * cos_az
        
        samples[i] = [x3, y3, vx3, vy3]
    
    return samples


def sample_satellite_state_near_planet(
    Y_sp0: np.ndarray,
    N: int = 1,
    R_p: Optional[float] = None,
    r_min_factor: Optional[float] = None,
    r_max_factor: Optional[float] = None,
    v_rel_min: Optional[float] = None,
    v_rel_max: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample N satellite states around the planet at t=0.
    Uses planet-relative (Cartesian) coordinates.
    
    Parameters
    ----------
    Y_sp0 : np.ndarray
        Star+planet state [xs, ys, vxs, vys, xp, yp, vxp, vyp]
    N : int
        Number of samples
    R_p : float, optional
        Planet radius (km). If None, uses Jupiter radius.
    r_min_factor : float, optional
        Min radius in R_p units (e.g., 20 → 20*R_p)
    r_max_factor : float, optional
        Max radius in R_p units
    v_rel_min : float, optional
        Min relative velocity (km/s)
    v_rel_max : float, optional
        Max relative velocity (km/s)
    rng : np.random.Generator, optional
        Random number generator
    
    Returns
    -------
    np.ndarray
        Shape (N, 4): [x3, y3, vx3, vy3] in barycentric frame
    """
    if R_p is None:
        R_p = R_JUP
    if r_min_factor is None:
        r_min_factor = 20.0
    if r_max_factor is None:
        r_max_factor = 500.0
    if v_rel_min is None:
        v_rel_min = 12.0
    if v_rel_max is None:
        v_rel_max = 80.0
    if rng is None:
        rng = np.random.default_rng()
    
    xs, ys, vxs, vys, xp, yp, vxp, vyp = Y_sp0
    
    samples = np.zeros((N, 4))
    
    for i in range(N):
        # Radius from planet center
        r0 = R_p * rng.uniform(r_min_factor, r_max_factor)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        
        # Unit radial vector
        ex = np.cos(theta)
        ey = np.sin(theta)
        
        # Position relative to planet
        x3 = xp + r0 * ex
        y3 = yp + r0 * ey
        
        # Relative velocity magnitude
        v_rel_mag = rng.uniform(v_rel_min, v_rel_max)
        
        # Direction: mostly inward with some tangential spread
        alpha = rng.uniform(-np.pi / 4.0, np.pi / 4.0)
        
        # Radial inward (-e_r) and tangential (+e_theta) basis
        er_in = -np.array([ex, ey])
        et = np.array([-ey, ex])  # 90° counter-clockwise
        
        # Relative velocity direction
        dir_vec = np.cos(alpha) * er_in + np.sin(alpha) * et
        dir_vec /= np.linalg.norm(dir_vec)
        
        v_rel = v_rel_mag * dir_vec
        
        # Barycentric velocity
        vx3 = vxp + v_rel[0]
        vy3 = vyp + v_rel[1]
        
        samples[i] = [x3, y3, vx3, vy3]
    
    return samples
