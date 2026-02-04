import numpy as np

def norm(v):
    return float(np.hypot(v[0], v[1]))

def sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

def rot2(vhat, theta, s):
    c = np.cos(theta)
    sn = np.sin(theta)
    x, y = vhat
    return np.array([ c*x - s*sn*y, s*sn*x + c*y ], dtype=float)

def flyby_unpowered(xm0, ym0, um0, vm0, vstar0, M, G):
    mu = G*M

    r0 = np.array([xm0, ym0], dtype=float)
    v0 = np.array([um0, vm0 - vstar0], dtype=float)

    r0mag = norm(r0)
    v0mag = norm(v0)

    eps = 0.5*v0mag**2 - mu/r0mag
    if eps <= 0:
        raise ValueError(f"Bound/parabolic: eps={eps} <= 0 (hyperbola requires eps>0).")

    vinf = np.sqrt(2*eps)

    h = r0[0]*v0[1] - r0[1]*v0[0]          # signed
    s = sign(h)

    b = abs(h)/vinf
    e = np.sqrt(1 + (b*vinf**2/mu)**2)
    a = -mu/(vinf**2)

    # deflection angle
    theta = 2*np.arctan2(mu, b*vinf**2)    # equivalent to 2*arcsin(1/e)

    # perihelion distance and speed (star frame)
    rp = a*(1 - e**2)/(1 + e)
    vp = np.sqrt(vinf**2 + 2*mu/rp)

    # outgoing asymptotic velocity in star frame
    vhat_in = v0 / v0mag
    vhat_out = rot2(vhat_in, theta, s)
    vrel_out = vinf * vhat_out

    # lab frame
    vlab_out = np.array([vrel_out[0], vrel_out[1] + vstar0], dtype=float)
    vlab_in  = np.array([um0, vm0], dtype=float)

    dV = norm(vlab_out) - norm(vlab_in)

    return {
        "eps": eps,
        "vinf": vinf,
        "h": h,
        "s": s,
        "b": b,
        "e": e,
        "a": a,
        "theta": theta,
        "rp": rp,
        "vp": vp,
        "umF": vlab_out[0],
        "vmF": vlab_out[1],
        "dV": dV,
    }

def flyby_oberth(xm0, ym0, um0, vm0, vstar0, M, G, dv_peri):
    """
    Oberth model:
      - Incoming hyperbola gives e1, rp, vp1 and half-angle δ1 = asin(1/e1)
      - At perihelion apply tangential dv_peri: vp2 = vp1 + dv
      - New outbound hyperbola has eps2, vinf2, h2, e2 and half-angle δ2 = asin(1/e2)
      - Total deflection relative to incoming asymptote: theta_tot = δ1 + δ2
      - Outgoing asymptote direction = rotate incoming vhat by theta_tot (same sign s)
      - Outgoing speed at infinity = vinf2
    """
    mu = G*M

    # First compute incoming hyperbola invariants + perihelion
    base = flyby_unpowered(xm0, ym0, um0, vm0, vstar0, M, G)

    e1 = base["e"]
    rp = base["rp"]
    vp1 = base["vp"]
    s = base["s"]

    delta1 = np.arcsin(1.0/e1)

    vp2 = vp1 + dv_peri

    eps2 = 0.5*vp2**2 - mu/rp
    if eps2 <= 0:
        raise ValueError(f"Post-burn bound/parabolic: eps2={eps2} <= 0 (need eps2>0).")

    vinf2 = np.sqrt(2*eps2)
    h2 = rp*vp2
    e2 = np.sqrt(1 + (2*eps2*h2**2)/(mu**2))
    delta2 = np.arcsin(1.0/e2)

    theta_tot = delta1 + delta2

    # incoming direction (star frame)
    v0 = np.array([um0, vm0 - vstar0], dtype=float)
    v0mag = norm(v0)
    vhat_in = v0 / v0mag

    # outgoing asymptote in star frame
    vhat_out = rot2(vhat_in, theta_tot, s)
    vrel_out = vinf2 * vhat_out

    # lab frame
    vlab_out = np.array([vrel_out[0], vrel_out[1] + vstar0], dtype=float)
    vlab_in  = np.array([um0, vm0], dtype=float)

    dV = norm(vlab_out) - norm(vlab_in)

    return {
        "rp": rp,
        "vp1": vp1,
        "vp2": vp2,
        "eps2": eps2,
        "vinf2": vinf2,
        "h2": h2,
        "e2": e2,
        "theta_tot": theta_tot,
        "umF": vlab_out[0],
        "vmF": vlab_out[1],
        "dV": dV,
        # also return incoming baseline if useful
        "baseline": base
    }
