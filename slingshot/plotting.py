"""
Visualization and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


M_SUN = 1.98847e30  # kg
R_SUN = 696000.0  # km
M_JUP = 1.898e27  # kg
R_JUP = 71492.0  # km


def plot_best_candidate_with_bodies(
    sol: object,
    analysis: Optional[Dict[str, Any]] = None,
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
    R_star: Optional[float] = None,
    R_p: Optional[float] = None,
    figsize: tuple = (10, 10),
    title: str = "Slingshot Trajectory",
) -> plt.Figure:
    """
    Plot trajectory with scaled star and planet circles.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    analysis : dict, optional
        Analysis results to display
    m_star, m_p : float, optional
        Masses for radius scaling
    R_star, R_p : float, optional
        Radii (km)
    figsize : tuple
        Figure size
    title : str
        Plot title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if m_star is None:
        m_star = M_SUN
    if m_p is None:
        m_p = M_JUP
    if R_p is None:
        R_p = R_JUP
    
    # Estimate star radius if not provided
    if R_star is None:
        R_star = R_SUN * np.sqrt(m_star / M_SUN)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if sol is None:
        ax.text(0.5, 0.5, "No valid solution", ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return fig
    
    y = sol.y
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    x3 = y[8]
    y3 = y[9]
    
    # Plot trajectories
    ax.plot(xs, ys, label="Star", lw=1.5, alpha=0.7, color='gold')
    ax.plot(xp, yp, label="Planet", lw=1.5, alpha=0.7, color='darkred')
    ax.plot(x3, y3, label="Satellite", lw=2.0, alpha=0.8, color='blue')
    
    # Initial position circles
    star_circle = plt.Circle((xs[0], ys[0]), R_star, fill=True, facecolor='gold',
                             alpha=0.6, edgecolor='orange', linewidth=2)
    planet_circle = plt.Circle((xp[0], yp[0]), R_p, fill=True, facecolor='darkred',
                               alpha=0.6, edgecolor='red', linewidth=2)
    ax.add_artist(star_circle)
    ax.add_artist(planet_circle)
    
    # Final positions
    ax.plot(xs[-1], ys[-1], 'o', color='gold', markersize=8)
    ax.plot(xp[-1], yp[-1], 'o', color='darkred', markersize=8)
    ax.plot(x3[-1], y3[-1], 'o', color='blue', markersize=8)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (km)', fontsize=11)
    ax.set_ylabel('y (km)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Analysis text box
    if analysis:
        textlines = []
        if 'delta_v' in analysis:
            textlines.append(f"Δv = {analysis['delta_v']:.2f} km/s")
        if 'deflection' in analysis:
            textlines.append(f"Deflection = {analysis['deflection']:.1f}°")
        if 'r_min' in analysis:
            textlines.append(f"r_min = {analysis['r_min']:.0f} km")
        if 'unbound_f' in analysis:
            textlines.append(f"Unbound: {analysis['unbound_f']}")
        
        textstr = '\n'.join(textlines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    return fig


def plot_mc_summary(
    mc: Dict[str, Any],
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot Monte Carlo summary: scatter of deflection vs Δv, histogram of deflection/180°.
    
    Parameters
    ----------
    mc : dict
        Monte Carlo results
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    ok = mc["ok"]
    dv = mc["delta_v"][ok]
    df = mc["deflection"][ok]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter: deflection vs Δv
    ax = axes[0]
    ax.scatter(df, dv, s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel("Deflection angle (deg)", fontsize=11)
    ax.set_ylabel("Δv (km/s)", fontsize=11)
    ax.set_title("Slingshot outcomes (successful)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Histogram: deflection fraction
    ax = axes[1]
    deflection_frac = df / 180.0
    ax.hist(deflection_frac, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Deflection / 180°", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Deflection distribution", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_velocity_phase_space(
    sol: object,
    title_prefix: str = "",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot velocity phase space: (vx, vy) and (v_radial, v_normal).
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    title_prefix : str
        Prefix for plot titles
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sol is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No solution available", ha='center', va='center',
                transform=ax.transAxes)
        return fig
    
    y = sol.y
    
    # Barycentric satellite velocity
    vx_sat = y[10]
    vy_sat = y[11]
    
    # Planet-frame velocity
    vxp = y[6]
    vyp = y[7]
    dvx = vx_sat - vxp
    dvy = vy_sat - vyp
    
    # Radial and normal components (planet frame)
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    dx = xs - xp
    dy = ys - yp
    r = np.hypot(dx, dy)
    
    # Unit radial (planet -> satellite)
    erx = dx / (r + 1e-10)
    ery = dy / (r + 1e-10)
    
    # Unit normal (tangential)
    enx = -ery
    eny = erx
    
    v_rad = dvx * erx + dvy * ery
    v_norm = dvx * enx + dvy * eny
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # (vx, vy) phase space
    ax = axes[0]
    ax.plot(vx_sat, vy_sat, lw=1, alpha=0.8)
    ax.set_xlabel('vx (km/s)', fontsize=11)
    ax.set_ylabel('vy (km/s)', fontsize=11)
    ax.set_title(f'{title_prefix} Satellite (vx, vy) phase space', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # (v_radial, v_normal) phase space
    ax = axes[1]
    ax.plot(v_rad, v_norm, lw=1, alpha=0.8)
    ax.set_xlabel('v_radial (km/s)', fontsize=11)
    ax.set_ylabel('v_normal (km/s)', fontsize=11)
    ax.set_title(f'{title_prefix} Radial vs Normal velocity', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
