"""
Two-body gravity-assist wrapper module  (km-kg-s unit system).

Provides a clean interface to TwoBodyScatter for:
- Computing hyperbolic encounter trajectories
- Extracting orbital energy metrics
- Organising encounter geometry parameters

Supports scattering off **either** the star or the planet so that
trajectory_tracks.py can compute both baselines.

Unit contract (matches the rest of slingshot/):
    Distances : km
    Velocities: km/s
    Masses    : kg
    Energy    : km²/s²  (≡ MJ/kg)
    μ         : km³/s²
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any, Union, Sequence
import sys
from pathlib import Path

from .constants import G_KM, M_SUN, M_JUP, R_SUN, R_JUP

StarVelocity = Union[float, Sequence[float]]


def _star_velocity_components(vstar0: StarVelocity) -> Tuple[float, float]:
    """Resolve star velocity input to (vx, vy) with legacy scalar support."""
    if np.isscalar(vstar0):
        return 0.0, float(vstar0)
    if len(vstar0) != 2:
        raise ValueError(f"Star velocity vector must have 2 components, got {vstar0}")
    return float(vstar0[0]), float(vstar0[1])

# Import TwoBodyScatter (unit-agnostic closed-form solver)
try:
    from TwoBodyScatter import gravity_assist_no_burn, deltaV_lab
except ImportError:
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from TwoBodyScatter import gravity_assist_no_burn, deltaV_lab


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class TwoBodyGeometry:
    """Parameters defining a 2-body hyperbolic encounter (km-kg-s)."""

    # Initial state (lab frame)
    xm0: float          # Initial x position  [km]
    ym0: float          # Initial y position  [km]
    um0: float          # Initial x velocity  [km/s]
    vm0: float          # Initial y velocity  [km/s]

    # Environment
    vstar0: float       # Legacy star velocity y-component [km/s]
    vstar_vec: Tuple[float, float]  # Full star velocity vector [km/s]
    mu: float           # G·M of scattering body [km³/s²]

    # Derived
    impact_parameter: float   # b   [km]
    approach_angle: float     # radians


@dataclass
class TrajectoryResult:
    """Results from a single 2-body trajectory computation (km-kg-s)."""

    x_star: np.ndarray        # X along trajectory [km]
    y_star: np.ndarray        # Y along trajectory [km]

    orbital_energy: float     # ½ΔV² specific energy [km²/s²]
    deltaV: float             # ΔV magnitude (vector diff) [km/s]

    epsilon: float            # Specific orbital energy [km²/s²]
    vinf: float               # Asymptotic relative speed [km/s]
    e: float                  # Eccentricity
    a: float                  # Semi-major axis [km]
    rp: float = 0.0           # Periapsis distance [km]

    # Final lab-frame velocity components
    umF: float = 0.0          # Final x velocity [km/s]
    vmF: float = 0.0          # Final y velocity [km/s]

    geometry: Optional[TwoBodyGeometry] = None
    valid: bool = False


# -----------------------------------------------------------------------
# Encounter engine
# -----------------------------------------------------------------------

class TwoBodyEncounter:
    """Manager for 2-body hyperbolic encounters.

    All inputs/outputs are in **km-kg-s**.
    The underlying TwoBodyScatter functions are unit-agnostic —
    they just compute with whatever μ you pass.
    """

    def __init__(
        self,
        M_body_kg: float,
        G: float = G_KM,
        *,
        label: str = "star",
        R_body_km: float = 0.0,
    ):
        """
        Parameters
        ----------
        M_body_kg : float
            Mass of the scattering body [kg].
        G : float
            Gravitational constant [km³ kg⁻¹ s⁻²].
            Default is ``G_KM`` from ``slingshot.constants``.
        label : str
            Human label for this body ("star" or "planet").
        R_body_km : float
            Physical radius of the body [km].  Trajectories with
            periapsis rp < R_body_km are rejected as body-interior
            collisions.  Set to 0 to disable.
        """
        self.M_body_kg = M_body_kg
        self.G = G
        self.mu = G * M_body_kg   # km³/s²
        self.label = label
        self.R_body_km = R_body_km

    # ---- single encounter -------------------------------------------

    def compute_trajectory(
        self,
        xm0: float,
        ym0: float,
        um0: float,
        vm0: float,
        vstar0: StarVelocity,
        num_points: int = 400,
    ) -> TrajectoryResult:
        """Compute one hyperbolic encounter.

        Parameters
        ----------
        xm0, ym0 : float   Position [km]
        um0, vm0 : float   Velocity [km/s]
        vstar0   : float or (float, float)
            Star velocity input. Scalar keeps legacy behavior
            (interpreted as y-component only); tuple/list gives full
            (vx, vy) in km/s.
        num_points : int    Trajectory sample count.

        Returns
        -------
        TrajectoryResult
        """
        vstar_x, vstar_y = _star_velocity_components(vstar0)
        vstar_vec = (vstar_x, vstar_y)

        _empty = TrajectoryResult(
            x_star=np.array([]), y_star=np.array([]),
            orbital_energy=0.0, deltaV=0.0,
            epsilon=0.0, vinf=0.0, e=0.0, a=0.0,
            geometry=TwoBodyGeometry(
                xm0=xm0, ym0=ym0, um0=um0, vm0=vm0,
                vstar0=vstar_y, vstar_vec=vstar_vec, mu=self.mu,
                impact_parameter=0.0, approach_angle=0.0,
            ),
            valid=False,
        )

        try:
            result = gravity_assist_no_burn(xm0, ym0, um0, vm0, vstar_vec, self.mu)
            eps = result.epsilon
            vinf = result.vinf
            e = result.e
            a = result.a

            if e <= 1.0 or a >= 0 or eps <= 0:
                _empty.epsilon, _empty.vinf, _empty.e, _empty.a = eps, vinf, e, a
                return _empty

            # Reject if periapsis is inside the body
            rp = result.rp
            if self.R_body_km > 0 and rp < self.R_body_km:
                _empty.epsilon, _empty.vinf, _empty.e, _empty.a = eps, vinf, e, a
                _empty.rp = rp
                return _empty

            # Hyperbolic trajectory in star frame
            nu_inf = np.arccos(-1.0 / e)
            nu = np.linspace(-nu_inf + 0.01, nu_inf - 0.01, num_points)
            r = a * (e * e - 1.0) / (1.0 + e * np.cos(nu))

            vx_s = um0 - vstar_x
            vy_s = vm0 - vstar_y
            v_mag = np.hypot(vx_s, vy_s)
            if v_mag == 0:
                return _empty

            ex, ey = vx_s / v_mag, vy_s / v_mag
            epx, epy = -ey, ex

            x_star = r * (np.cos(nu) * ex + np.sin(nu) * epx)
            y_star = r * (np.cos(nu) * ey + np.sin(nu) * epy)

            dV = deltaV_lab(um0, vm0, result.umF, result.vmF)
            energy_specific = 0.5 * dV ** 2    # km²/s²

            hz = xm0 * vy_s - ym0 * vx_s
            b_mag = abs(hz) / v_mag if v_mag > 0 else 0.0
            angle = np.arctan2(vy_s, vx_s)

            return TrajectoryResult(
                x_star=x_star, y_star=y_star,
                orbital_energy=energy_specific,
                deltaV=dV,
                epsilon=eps, vinf=vinf, e=e, a=a,
                rp=result.rp,
                umF=result.umF, vmF=result.vmF,
                geometry=TwoBodyGeometry(
                    xm0=xm0, ym0=ym0, um0=um0, vm0=vm0,
                    vstar0=vstar_y, vstar_vec=vstar_vec, mu=self.mu,
                    impact_parameter=b_mag, approach_angle=angle,
                ),
                valid=True,
            )

        except (ValueError, ZeroDivisionError, AttributeError):
            return _empty

    # ---- parameter-space scan ----------------------------------------

    def scan_parameter_space(
        self,
        v_approach: float,
        vstar0: StarVelocity,
        r_start: float,
        b_values: np.ndarray,
        angle_values: np.ndarray,
        num_points: int = 400,
    ) -> Tuple[List[TrajectoryResult], List[float], np.ndarray]:
        """Scan impact parameter × approach angle grid.

        All inputs in km / km/s.

        Returns
        -------
        trajectories, energy_values, parameter_grid
        """
        trajectories: List[TrajectoryResult] = []
        energy_values: List[float] = []
        parameter_grid: List[Tuple[float, float]] = []
        vstar_x, vstar_y = _star_velocity_components(vstar0)
        vstar_vec = (vstar_x, vstar_y)

        total = len(b_values) * len(angle_values)
        print(f"[{self.label}] Scanning {len(b_values)} × {len(angle_values)} = {total} encounters …")

        valid_count = 0
        count = 0
        for angle in angle_values:
            vx = v_approach * np.cos(angle)
            vy = v_approach * np.sin(angle)
            for b_mag in b_values:
                perp = angle + np.pi / 2
                xm0 = -vx / v_approach * r_start + b_mag * np.cos(perp)
                ym0 = -vy / v_approach * r_start + b_mag * np.sin(perp)
                um0 = vx + vstar_x
                vm0 = vy + vstar_y

                traj = self.compute_trajectory(xm0, ym0, um0, vm0, vstar_vec, num_points)
                if traj.valid:
                    trajectories.append(traj)
                    energy_values.append(traj.orbital_energy)
                    parameter_grid.append((b_mag, angle))
                    valid_count += 1
                count += 1
                if count % max(1, total // 10) == 0:
                    print(f"  {count}/{total} ({valid_count} valid) …")

        print(f"[{self.label}] Complete: {valid_count}/{total} successful")
        return trajectories, energy_values, np.array(parameter_grid)

    # ---- statistics --------------------------------------------------

    @staticmethod
    def get_energy_statistics(energy_values: List[float]) -> Dict[str, float]:
        """Compute statistics on orbital energy distribution.

        Input/output in km²/s² (≡ MJ/kg).
        """
        e = np.asarray(energy_values)
        return {
            "min": float(np.min(e)),
            "max": float(np.max(e)),
            "mean": float(np.mean(e)),
            "median": float(np.median(e)),
            "std": float(np.std(e)),
            "count": len(e),
        }


# -----------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------

def create_encounter_from_config(cfg) -> TwoBodyEncounter:
    """Create a TwoBodyEncounter from a FullConfig object (star scattering).

    Parameters
    ----------
    cfg : FullConfig | dict
        If dict, expects flat ``system.M_star_Msun`` key.
    """
    if hasattr(cfg, "system"):
        M_star_Msun = cfg.system.M_star_Msun
        R_star_Rsun = getattr(cfg.system, 'R_star_Rsun', 1.0)
    else:
        M_star_Msun = cfg.get("system", {}).get("M_star_Msun", 1.19)
        R_star_Rsun = cfg.get("system", {}).get("R_star_Rsun", 1.0)

    return TwoBodyEncounter(M_star_Msun * M_SUN, G_KM, label="star",
                            R_body_km=R_star_Rsun * R_SUN)


def create_planet_encounter_from_config(cfg) -> TwoBodyEncounter:
    """Create a TwoBodyEncounter for *planet* scattering from a FullConfig.

    Parameters
    ----------
    cfg : FullConfig | dict
    """
    if hasattr(cfg, "system"):
        M_planet_Mjup = cfg.system.M_planet_Mjup
        R_planet_Rjup = getattr(cfg.system, 'R_planet_Rjup', 1.155)
    else:
        M_planet_Mjup = cfg.get("system", {}).get("M_planet_Mjup", 5.2)
        R_planet_Rjup = cfg.get("system", {}).get("R_planet_Rjup", 1.155)

    return TwoBodyEncounter(M_planet_Mjup * M_JUP, G_KM, label="planet",
                            R_body_km=R_planet_Rjup * R_JUP)
