"""
Output layer for slingshot-solver.

Submodules:
    plotting         — 3-body diagnostic plots
    plotting_twobody — 2-body parameter-space heatmaps & trajectory tracks
    animation        — Matplotlib FuncAnimation video rendering
    report           — Auto-generated REPORT.md
    compare_runs     — Cross-run comparison from summary.csv
"""

from .plotting import (
    plot_best_candidate_with_bodies,
    plot_mc_summary,
    plot_velocity_phase_space,
    plot_star_proximity_distribution,
    plot_planet_frame_diagnostics,
    plot_multi_candidate_overlay,
    plot_rejection_breakdown,
    plot_parameter_correlations,
    plot_energy_cdf,
)

from .plotting_twobody import (
    plot_poincare_heatmaps,
    plot_scattering_maps,
    plot_encounter_2d_cartesian,
    plot_encounter_2d_trajectories,
    plot_oberth_comparison,
    plot_trajectory_tracks,
)

from .animation import (
    animate_trajectory,
    animate_phase_space,
    generate_all_animations,
)

from .report import generate_run_report
from .compare_runs import compare_runs
