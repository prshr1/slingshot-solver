"""
Closed-form 2D gravity-assist (hyperbolic scattering) in a moving-star frame
+ optional time-of-periapsis (tb) via hyperbolic anomaly.

This is a direct Python translation of the mechanics you wrote:
- Transform to star frame
- Compute invariants (epsilon, h, e, b, theta, etc.)
- Rotate v_infty(-) by the scattering angle to get v_infty(+)
- Transform back to lab frame

Conventions:
- Star velocity is (0, vstar0, 0)
- Particle initial lab state at t0: r0=(xm0, ym0, 0), V0=(um0, vm0, 0)
- Work in xy plane; z=0 always
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple


# ---------------------------
# Helpers (scalar math)
# ---------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def norm2(x: float, y: float) -> float:
    return math.hypot(x, y)

def sgn(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

def acosh(x: float) -> float:
    # Python's math.acosh exists, but keep as explicit alias.
    return math.acosh(x)


# ---------------------------
# Star-frame primitives
# ---------------------------

def r0fn(xm0: float, ym0: float) -> float:
    """Initial separation magnitude r0 = ||(xm0, ym0)||."""
    return norm2(xm0, ym0)

def v0_vec_star(um0: float, vm0: float, vstar0: float) -> Tuple[float, float]:
    """Relative velocity vector in star frame."""
    return (um0, vm0 - vstar0)

def v0fn(um0: float, vm0: float, vstar0: float) -> float:
    """Initial relative speed in star frame v0 = ||(um0, vm0-vstar0)||."""
    vx, vy = v0_vec_star(um0, vm0, vstar0)
    return norm2(vx, vy)

def hzfn(xm0: float, ym0: float, um0: float, vm0: float, vstar0: float) -> float:
    """Planar specific angular momentum z-component: hz = x*vy - y*vx in star frame."""
    vx, vy = v0_vec_star(um0, vm0, vstar0)
    return xm0 * vy - ym0 * vx

def hfn(xm0: float, ym0: float, um0: float, vm0: float, vstar0: float) -> float:
    """Magnitude of planar specific angular momentum."""
    return abs(hzfn(xm0, ym0, um0, vm0, vstar0))

def epsilonfn(um0: float, vm0: float, vstar0: float, mu: float, xm0: float, ym0: float) -> float:
    """
    Specific orbital energy epsilon = v^2/2 - mu/r evaluated at t0 in star frame.
    NOTE: In the strict t0 -> -infty limit, r->infty, so epsilon -> v_inf^2/2.
    """
    r0 = r0fn(xm0, ym0)
    v0 = v0fn(um0, vm0, vstar0)
    return 0.5 * v0 * v0 - mu / r0

def vinffn(epsilon: float) -> float:
    """Asymptotic speed v_inf = sqrt(2 epsilon). Requires epsilon > 0 (hyperbola)."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0 for hyperbola; got {epsilon}")
    return math.sqrt(2.0 * epsilon)

def bfn(xm0: float, ym0: float, um0: float, vm0: float, vstar0: float) -> float:
    """Impact parameter b = |h|/v0."""
    v0 = v0fn(um0, vm0, vstar0)
    if v0 == 0:
        raise ValueError("b undefined: v0=0")
    return abs(hzfn(xm0, ym0, um0, vm0, vstar0)) / v0


# ---------------------------
# Hyperbola geometry
# ---------------------------

def e_from_b_vinf(b: float, vinf: float, mu: float) -> float:
    """Eccentricity e = sqrt(1 + (b*vinf^2/mu)^2)."""
    if mu <= 0:
        raise ValueError("mu must be positive.")
    return math.sqrt(1.0 + (b * vinf * vinf / mu) ** 2)

def a_from_vinf(mu: float, vinf: float) -> float:
    """
    Semi-major axis for a hyperbola: a = -mu/(2 epsilon).
    With epsilon = vinf^2/2, this simplifies to a = -mu/vinf^2.
    """
    if vinf <= 0:
        raise ValueError("vinf must be positive.")
    return -mu / (vinf * vinf)

def rp_from_a_e(a: float, e: float) -> float:
    """Periapsis distance rp = |a|(e-1) = -a*(e-1) since a<0."""
    if e <= 1:
        raise ValueError("Hyperbola requires e>1.")
    return (-a) * (e - 1.0)

def theta_from_b_vinf(mu: float, b: float, vinf: float) -> float:
    """Deflection angle theta = 2*atan(mu/(b*vinf^2))."""
    if b <= 0:
        raise ValueError("b must be positive.")
    return 2.0 * math.atan(mu / (b * vinf * vinf))

def delta_from_e(e: float) -> float:
    """Half-deflection angle delta = arcsin(1/e)."""
    if e <= 1.0:
        raise ValueError("Hyperbola requires e>1 for delta.")
    return math.asin(_clamp(1.0 / e, -1.0, 1.0))


# ---------------------------
# Periapsis speed (needed for Oberth)
# ---------------------------

def vp_from_vinf_rp(mu: float, vinf: float, rp: float) -> float:
    """Speed at periapsis: vp^2 = vinf^2 + 2mu/rp."""
    if rp <= 0:
        raise ValueError("rp must be > 0")
    return math.sqrt(vinf * vinf + 2.0 * mu / rp)


# ---------------------------
# Time of periapsis passage (optional)
# ---------------------------

def rdot0fn(xm0: float, ym0: float, um0: float, vm0: float, vstar0: float) -> float:
    """Radial velocity at t0 in star frame: rdot = (r·v)/|r|."""
    r = r0fn(xm0, ym0)
    if r == 0:
        raise ValueError("rdot undefined: r0=0.")
    vx = um0
    vy = vm0 - vstar0
    return (xm0 * vx + ym0 * vy) / r

def sinf0fn(mu: float, e: float, h: float, rdot0: float) -> float:
    """sin f0 = (h * rdot0) / (mu * e)."""
    if mu <= 0 or e <= 0:
        raise ValueError("mu and e must be positive.")
    return (h * rdot0) / (mu * e)

def cosf0fn(a: float, e: float, r0: float) -> float:
    """
    From r = a(e^2-1)/(1+e cos f) for hyperbola with a<0.
    Solve: cos f = (a(e^2-1)/r - 1)/e
    """
    if r0 <= 0:
        raise ValueError("r0 must be positive.")
    return (a * (e * e - 1.0) / r0 - 1.0) / e

def H0fn(e: float, f0: float) -> float:
    """H0 = arcosh((e+cos f0)/(1+e cos f0))."""
    denom = 1.0 + e * math.cos(f0)
    if denom <= 0:
        raise ValueError("Invalid geometry: 1+e cos(f0) must be > 0 for this formula.")
    arg = (e + math.cos(f0)) / denom
    if arg < 1.0:
        # acosh domain requires >=1; numerical issues can push slightly below 1
        arg = 1.0
    return acosh(arg)

def mean_motion_hyperbola(mu: float, a: float) -> float:
    """n = sqrt(mu/(-a)^3) for hyperbola (a<0)."""
    if a >= 0:
        raise ValueError("Hyperbola requires a<0.")
    return math.sqrt(mu / ((-a) ** 3))

def tbfn(t0: float, e: float, a: float, H0: float, mu: float) -> float:
    """tb = t0 - (e*sinh(H0) - H0)/n."""
    n = mean_motion_hyperbola(mu, a)
    return t0 - (e * math.sinh(H0) - H0) / n


# ---------------------------
# Vector rotation to get v_infty(+)
# ---------------------------

def vout_star_frame_from_theta(
    um0: float, vm0: float, vstar0: float, theta: float, hz: float, vmag_out: float
) -> Tuple[float, float]:
    """
    Build orthonormal basis e1 along incoming v(-), e2 = hhat x e1,
    then v_out = vmag_out*(cos(theta)*e1 + sin(theta)*e2).
    """
    vx_in, vy_in = v0_vec_star(um0, vm0, vstar0)
    vin = norm2(vx_in, vy_in)
    if vin == 0:
        raise ValueError("Cannot rotate: incoming star-frame speed is 0")

    e1x, e1y = vx_in / vin, vy_in / vin
    s = sgn(hz)
    # zhat x (e1x,e1y,0) = (-e1y, e1x, 0); multiply by sign(hz)
    e2x, e2y = s * (-e1y), s * (e1x)

    vx_out = vmag_out * (math.cos(theta) * e1x + math.sin(theta) * e2x)
    vy_out = vmag_out * (math.cos(theta) * e1y + math.sin(theta) * e2y)
    return vx_out, vy_out

def lab_from_star(vx_s: float, vy_s: float, vstar0: float) -> Tuple[float, float, float]:
    """Transform velocity from star frame to lab frame."""
    return (vx_s, vy_s + vstar0, 0.0)


# ---------------------------
# No-burn (single hyperbola) solution
# ---------------------------

@dataclass
class NoBurnResult:
    epsilon: float
    vinf: float
    b: float
    e: float
    a: float
    rp: float
    theta: float
    umF: float
    vmF: float
    wmF: float

def gravity_assist_no_burn(
    xm0: float, ym0: float,
    um0: float, vm0: float,
    vstar0: float,
    mu: float
) -> NoBurnResult:
    """Compute gravity assist with no burn (single hyperbola)."""
    eps = epsilonfn(um0, vm0, vstar0, mu, xm0, ym0)
    vinf = vinffn(eps)
    b = bfn(xm0, ym0, um0, vm0, vstar0)
    e = e_from_b_vinf(b, vinf, mu)
    a = a_from_vinf(mu, vinf)
    rp = rp_from_a_e(a, e)
    theta = theta_from_b_vinf(mu, b, vinf)

    hz = hzfn(xm0, ym0, um0, vm0, vstar0)
    vx_out_s, vy_out_s = vout_star_frame_from_theta(um0, vm0, vstar0, theta, hz, vmag_out=vinf)
    umF, vmF, wmF = lab_from_star(vx_out_s, vy_out_s, vstar0)

    return NoBurnResult(eps, vinf, b, e, a, rp, theta, umF, vmF, wmF)


# ---------------------------
# Oberth-powered (two-hyperbola) solution
# ---------------------------

@dataclass
class OberthResult:
    # pre-burn hyperbola
    epsilon1: float
    vinf1: float
    b: float
    e1: float
    a1: float
    rp: float
    vp1: float
    delta1: float
    # post-burn hyperbola
    dv: float
    vp2: float
    epsilon2: float
    vinf2: float
    h2: float
    e2: float
    delta2: float
    theta_total: float
    # final lab velocity
    umF: float
    vmF: float
    wmF: float

def gravity_assist_oberth(
    xm0: float, ym0: float,
    um0: float, vm0: float,
    vstar0: float,
    mu: float,
    dv: float
) -> OberthResult:
    """Compute gravity assist with Oberth burn at periapsis (two hyperbolas)."""
    # --- hyperbola 1 (incoming to periapsis) ---
    eps1 = epsilonfn(um0, vm0, vstar0, mu, xm0, ym0)
    vinf1 = vinffn(eps1)
    b = bfn(xm0, ym0, um0, vm0, vstar0)
    e1 = e_from_b_vinf(b, vinf1, mu)
    a1 = a_from_vinf(mu, vinf1)
    rp = rp_from_a_e(a1, e1)

    vp1 = vp_from_vinf_rp(mu, vinf1, rp)
    delta1 = delta_from_e(e1)

    # --- Oberth burn at periapsis: tangential dv along direction of motion ---
    vp2 = vp1 + dv

    # new invariants for hyperbola 2
    eps2 = 0.5 * vp2 * vp2 - mu / rp
    vinf2 = vinffn(eps2)  # must remain >0, else you became bound
    h2 = rp * vp2
    e2 = math.sqrt(1.0 + (2.0 * eps2 * h2 * h2) / (mu * mu))
    delta2 = delta_from_e(e2)

    theta_total = delta1 + delta2

    # outgoing velocity at +infty is rotation of incoming direction by theta_total,
    # but with magnitude vinf2 (post-burn asymptotic speed).
    hz = hzfn(xm0, ym0, um0, vm0, vstar0)
    vx_out_s, vy_out_s = vout_star_frame_from_theta(
        um0, vm0, vstar0, theta_total, hz, vmag_out=vinf2
    )
    umF, vmF, wmF = lab_from_star(vx_out_s, vy_out_s, vstar0)

    return OberthResult(
        epsilon1=eps1, vinf1=vinf1, b=b, e1=e1, a1=a1, rp=rp, vp1=vp1, delta1=delta1,
        dv=dv, vp2=vp2, epsilon2=eps2, vinf2=vinf2, h2=h2, e2=e2, delta2=delta2,
        theta_total=theta_total,
        umF=umF, vmF=vmF, wmF=wmF
    )


# ---------------------------
# Convenience: simple closed-form result (legacy)
# ---------------------------

@dataclass
class ScatteringResult:
    epsilon: float
    vinf: float
    hz: float
    h: float
    b: float
    e: float
    a: float
    rp: float
    theta: float
    umF: float
    vmF: float
    wmF: float

def gravity_assist_closed_form(
    xm0: float, ym0: float,
    um0: float, vm0: float,
    vstar0: float,
    mu: float
) -> ScatteringResult:
    """Full closed-form gravity assist (alias for no-burn with extra info)."""
    eps = epsilonfn(um0, vm0, vstar0, mu, xm0, ym0)
    vinf = vinffn(eps)

    hz = hzfn(xm0, ym0, um0, vm0, vstar0)
    h = abs(hz)
    b = h / vinf

    e = e_from_b_vinf(b, vinf, mu)
    a = a_from_vinf(mu, vinf)
    rp = rp_from_a_e(a, e)

    theta = theta_from_b_vinf(mu, b, vinf)
    vx_out_s, vy_out_s = vout_star_frame_from_theta(um0, vm0, vstar0, theta, hz, vmag_out=vinf)

    umF = vx_out_s
    vmF = vy_out_s + vstar0
    wmF = 0.0

    return ScatteringResult(
        epsilon=eps, vinf=vinf, hz=hz, h=h, b=b, e=e, a=a, rp=rp, theta=theta,
        umF=umF, vmF=vmF, wmF=wmF
    )


# ---------------------------
# Helper: compute DeltaV in lab frame
# ---------------------------

def deltaV_lab(um0: float, vm0: float, umF: float, vmF: float) -> float:
    """Magnitude of change in lab-frame velocity vector."""
    return norm2(umF - um0, vmF - vm0)


# ---------------------------
# Example usage and testing
# ---------------------------

if __name__ == "__main__":
    # Example values (consistent units: SI-like, but relative orbital dynamics)
    G = 6.674e-11
    M = 1.989e30
    mu = G * M

    # Parameters for a hyperbolic encounter: object must have escape velocity
    # Higher r0 and sufficient v0 to ensure epsilon > 0
    xm0, ym0 = 5.0e12, 1.0e12  # Farther out initial position
    um0, vm0 = 5.0e4, 3.0e4    # Higher velocity in star frame (hyperbolic)
    vstar0 = 1.0e4             # Star velocity

    # No burn
    nb = gravity_assist_no_burn(xm0, ym0, um0, vm0, vstar0, mu)
    dV_no = deltaV_lab(um0, vm0, nb.umF, nb.vmF)

    # With Oberth at periapsis
    dv_peri = 5000.0  # m/s, for example
    ob = gravity_assist_oberth(xm0, ym0, um0, vm0, vstar0, mu, dv=dv_peri)
    dV_ob = deltaV_lab(um0, vm0, ob.umF, ob.vmF)

    print("\n--- No burn (single hyperbola) ---")
    print(f"epsilon = {nb.epsilon:.4e} J/kg")
    print(f"v_inf = {nb.vinf:.4e} m/s")
    print(f"impact parameter b = {nb.b:.4e} m")
    print(f"deflection theta = {nb.theta:.6f} rad ({nb.theta*180/3.14159:.2f}°)")
    print(f"periapsis rp = {nb.rp:.4e} m")
    print(f"umF, vmF = {nb.umF:.6e}, {nb.vmF:.6e} m/s")
    print(f"DeltaV_lab = {dV_no:.6e} m/s")

    print("\n--- With Oberth burn at periapsis ---")
    print(f"delta1 = {ob.delta1:.6f} rad, delta2 = {ob.delta2:.6f} rad")
    print(f"theta_total = {ob.theta_total:.6f} rad ({ob.theta_total*180/3.14159:.2f}°)")
    print(f"v_inf (pre-burn) = {ob.vinf1:.4e} m/s")
    print(f"v_inf (post-burn) = {ob.vinf2:.4e} m/s")
    print(f"Oberth dv = {ob.dv:.4e} m/s")
    print(f"umF, vmF = {ob.umF:.6e}, {ob.vmF:.6e} m/s")
    print(f"DeltaV_lab = {dV_ob:.6e} m/s")

    print("\n--- Oberth Effect Gain ---")
    print(f"DeltaV_lab (Oberth) - DeltaV_lab (no burn) = {(dV_ob - dV_no):.6e} m/s")
