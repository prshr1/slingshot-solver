"""
Animation and video rendering for trajectories.
Supports multiple animation types: trajectory, phase-space, comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path
from typing import Optional, Dict, Any, List


def animate_trajectory(
    sol: object,
    output_dir: str = "./results/frames",
    output_format: str = "mp4",
    fps: int = 30,
    figsize: tuple = (10, 10),
    show_bodies: bool = True,
    R_p: float = 71492.0,
    m_star: Optional[float] = None,
    m_p: Optional[float] = None,
) -> str:
    """
    Animate single slingshot trajectory with star/planet/satellite paths.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    output_dir : str
        Directory to save output
    output_format : str
        "mp4" or "gif"
    fps : int
        Frames per second
    figsize : tuple
        Figure size (inches)
    show_bodies : bool
        Show star/planet as circles
    R_p : float
        Planet radius (km)
    m_star : float, optional
        Star mass for radius scaling
    m_p : float, optional
        Planet mass (unused in this context)
    
    Returns
    -------
    str
        Path to output video file
    """
    from matplotlib.animation import PillowWriter, FFMpegWriter
    
    if sol is None:
        raise ValueError("Solution object is None")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    y = sol.y
    t = sol.t
    
    # Extract trajectories
    xs, ys = y[0], y[1]
    xp, yp = y[4], y[5]
    x3, y3 = y[8], y[9]
    
    # Determine plot limits
    all_x = np.concatenate([xs, xp, x3])
    all_y = np.concatenate([ys, yp, y3])
    xlim = [np.min(all_x) * 1.1, np.max(all_x) * 1.1]
    ylim = [np.min(all_y) * 1.1, np.max(all_y) * 1.1]
    
    # Figure setup
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial trajectory lines (to accumulate as animation progresses)
    line_star, = ax.plot([], [], lw=1.5, alpha=0.6, color='gold', label='Star')
    line_planet, = ax.plot([], [], lw=1.5, alpha=0.6, color='darkred', label='Planet')
    line_sat, = ax.plot([], [], lw=2.0, alpha=0.8, color='blue', label='Satellite')
    
    # Current position markers
    scat_star = ax.scatter([], [], s=60, c='gold', edgecolors='orange', linewidth=2, zorder=5)
    scat_planet = ax.scatter([], [], s=100, c='darkred', edgecolors='red', linewidth=2, zorder=5)
    scat_sat = ax.scatter([], [], s=40, c='blue', edgecolors='darkblue', linewidth=1.5, zorder=5)
    
    # Static bodies (initial positions)
    if show_bodies:
        circle_star = Circle((xs[0], ys[0]), R_p * 2, fill=False, edgecolor='orange',
                             linewidth=1, linestyle='--', alpha=0.5)
        circle_planet = Circle((xp[0], yp[0]), R_p, fill=True, facecolor='darkred',
                               alpha=0.3, edgecolor='red', linewidth=1)
        ax.add_artist(circle_star)
        ax.add_artist(circle_planet)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x (km)', fontsize=11)
    ax.set_ylabel('y (km)', fontsize=11)
    ax.set_title('Slingshot Trajectory Animation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=10)
    
    # Time and index text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        family='monospace')
    
    n_frames = len(t)
    
    def init():
        line_star.set_data([], [])
        line_planet.set_data([], [])
        line_sat.set_data([], [])
        scat_star.set_offsets(np.empty((0, 2)))
        scat_planet.set_offsets(np.empty((0, 2)))
        scat_sat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return line_star, line_planet, line_sat, scat_star, scat_planet, scat_sat, time_text
    
    def animate(frame):
        # Trailing path (last 100 points or all if fewer)
        start_idx = max(0, frame - 100)
        
        line_star.set_data(xs[start_idx:frame+1], ys[start_idx:frame+1])
        line_planet.set_data(xp[start_idx:frame+1], yp[start_idx:frame+1])
        line_sat.set_data(x3[start_idx:frame+1], y3[start_idx:frame+1])
        
        # Current position
        scat_star.set_offsets([[xs[frame], ys[frame]]])
        scat_planet.set_offsets([[xp[frame], yp[frame]]])
        scat_sat.set_offsets([[x3[frame], y3[frame]]])
        
        time_text.set_text(f"Time: {t[frame]:.1e} s\nFrame: {frame+1}/{n_frames}")
        
        return line_star, line_planet, line_sat, scat_star, scat_planet, scat_sat, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                    interval=1000//fps, blit=True, repeat=False)
    
    # Save
    output_file = Path(output_dir) / f"trajectory_animation.{output_format}"
    
    try:
        if output_format.lower() == 'mp4':
            writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
            anim.save(str(output_file), writer=writer, dpi=100)
        elif output_format.lower() == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(str(output_file), writer=writer, dpi=100)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    finally:
        plt.close(fig)
    
    return str(output_file)


def animate_phase_space(
    sol: object,
    output_dir: str = "./results/frames",
    output_format: str = "mp4",
    fps: int = 30,
) -> str:
    """
    Animate velocity phase space evolution (v_radial, v_normal).
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    output_dir : str
        Output directory
    output_format : str
        "mp4" or "gif"
    fps : int
        Frames per second
    
    Returns
    -------
    str
        Path to output file
    """
    from matplotlib.animation import PillowWriter, FFMpegWriter
    
    if sol is None:
        raise ValueError("Solution object is None")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    y = sol.y
    
    # Planet-frame velocity
    vxp = y[6]
    vyp = y[7]
    vx_sat = y[10]
    vy_sat = y[11]
    dvx = vx_sat - vxp
    dvy = vy_sat - vyp
    
    # Radial and normal components
    xs = y[0]
    ys = y[1]
    xp = y[4]
    yp = y[5]
    dx = xs - xp
    dy = ys - yp
    r = np.hypot(dx, dy)
    
    erx = dx / (r + 1e-10)
    ery = dy / (r + 1e-10)
    enx = -ery
    eny = erx
    
    v_rad = dvx * erx + dvy * ery
    v_norm = dvx * enx + dvy * eny
    
    n_frames = len(v_rad)
    
    # Limits
    vrad_lim = [np.min(v_rad) * 1.2, np.max(v_rad) * 1.2]
    vnorm_lim = [np.min(v_norm) * 1.2, np.max(v_norm) * 1.2]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    line, = ax.plot([], [], lw=2, alpha=0.7, color='blue')
    point, = ax.plot([], [], 'o', markersize=8, color='red', zorder=5)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        family='monospace')
    
    ax.set_xlim(vrad_lim)
    ax.set_ylim(vnorm_lim)
    ax.set_xlabel('v_radial (km/s)', fontsize=11)
    ax.set_ylabel('v_normal (km/s)', fontsize=11)
    ax.set_title('Velocity Phase Space Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text
    
    def animate(frame):
        start_idx = max(0, frame - 200)
        line.set_data(v_rad[start_idx:frame+1], v_norm[start_idx:frame+1])
        point.set_data([v_rad[frame]], [v_norm[frame]])
        time_text.set_text(f"Frame: {frame+1}/{n_frames}")
        return line, point, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                    interval=1000//fps, blit=True, repeat=False)
    
    output_file = Path(output_dir) / f"phasespace_animation.{output_format}"
    
    try:
        if output_format.lower() == 'mp4':
            writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=1500)
            anim.save(str(output_file), writer=writer, dpi=100)
        elif output_format.lower() == 'gif':
            writer = PillowWriter(fps=fps)
            anim.save(str(output_file), writer=writer, dpi=100)
    finally:
        plt.close(fig)
    
    return str(output_file)


def generate_all_animations(
    sol: object,
    output_dir: str = "./results/frames",
    video_fps: int = 30,
    video_format: str = "mp4",
    animate_trajectory: bool = True,
    animate_phase_space: bool = True,
) -> Dict[str, str]:
    """
    Generate all requested animation types.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Integrated trajectory
    output_dir : str
        Output directory
    video_fps : int
        Frames per second
    video_format : str
        "mp4" or "gif"
    animate_trajectory : bool
        Generate trajectory animation
    animate_phase_space : bool
        Generate phase-space animation
    
    Returns
    -------
    dict
        Mapping of animation type to output file path
    """
    results = {}
    
    try:
        if animate_trajectory:
            results['trajectory'] = animate_trajectory(
                sol, output_dir=output_dir, output_format=video_format, fps=video_fps
            )
    except Exception as e:
        print(f"Trajectory animation failed: {e}")
        results['trajectory'] = None
    
    try:
        if animate_phase_space:
            results['phase_space'] = animate_phase_space(
                sol, output_dir=output_dir, output_format=video_format, fps=video_fps
            )
    except Exception as e:
        print(f"Phase-space animation failed: {e}")
        results['phase_space'] = None
    
    return results
