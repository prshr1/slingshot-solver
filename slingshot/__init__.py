"""
Slingshot Solver: Gravitational slingshot dynamics in restricted 3-body systems

Package structure (v3.0.0):
    slingshot/
    ├── core/           — Physics engines + sampling
    │   ├── twobody_scatter  — Ground-truth closed-form 2-body scattering (pure math)
    │   ├── dynamics         — Restricted 3-body ODE integration
    │   ├── twobody          — OOP wrapper for TwoBodyScatter parameter-space scans
    │   └── sampling         — Initial condition generators
    ├── analysis/       — Trajectory analysis + comparisons
    │   ├── trajectory       — Encounter geometry extraction + metrics
    │   ├── monte_carlo      — MC batch processing, selection, filtering
    │   ├── baselines        — 2-body hyperbola + monopole baselines
    │   ├── narrowed_baselines — Envelope-matched 2-body sweeps
    │   └── comparison       — 2-body vs 3-body energy comparison
    ├── output/         — Visualization + reports + I/O
    │   ├── plotting         — 3-body diagnostic plots
    │   ├── plotting_twobody — 2-body parameter-space heatmaps
    │   ├── animation        — Video rendering
    │   ├── report           — Auto-generated REPORT.md
    │   └── compare_runs     — Cross-run comparison
    ├── config.py       — Pydantic config models + YAML I/O
    ├── constants.py    — Canonical km-kg-s physical constants
    ├── console.py      — Safe Unicode console output
    ├── pipeline.py     — 8-phase orchestrator
    └── cli.py          — CLI entry point
"""

__version__ = "3.0.0"

# ── Cross-cutting infrastructure ──────────────────────────────────────
from .constants import (
    G_KM,
    M_SUN,
    M_JUP,
    R_JUP,
    R_SUN,
    AU_KM,
    mu_star,
    mu_planet,
    au_to_km,
)

# ── Core: Physics engines + sampling ─────────────────────────────────
from .core.dynamics import (
    init_hot_jupiter_barycentric,
    restricted_3body_ode,
    simulate_3body,
)

from .core.sampling import (
    sample_satellite_state_barycentric,
    sample_satellite_state_near_planet,
)

from .core.twobody import (
    TwoBodyEncounter,
    TwoBodyGeometry,
    TrajectoryResult,
    create_encounter_from_config,
    create_planet_encounter_from_config,
)

# ── Analysis: trajectory analysis + comparisons ──────────────────────
from .analysis.trajectory import (
    analyze_trajectory,
    extract_encounter_states,
    EncounterGeometry,
)

from .analysis.monte_carlo import (
    evaluate_particle,
    run_monte_carlo,
    select_top_indices,
    select_pareto_indices,
    select_weighted_indices,
    resolve_metric_array,
)

from .analysis.baselines import (
    two_body_hyperbola_from_state,
    monopole_ode,
    simulate_monopole_baseline,
    compare_3body_with_baselines,
)

from .analysis.narrowed_baselines import (
    extract_envelope,
    compute_narrowed_baselines,
    run_narrowed_sweep,
    EnvelopeParams,
    NarrowedBaselineResult,
)

from .analysis.comparison import (
    compare_2body_3body,
    format_energy,
    print_comparison,
)

# ── Output: visualization + reports ──────────────────────────────────
from .output.plotting import (
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

from .output.plotting_twobody import (
    plot_poincare_heatmaps,
    plot_oberth_comparison,
    plot_trajectory_tracks,
)

from .output.animation import (
    animate_trajectory,
    animate_phase_space,
    generate_all_animations,
)

# ── Config ───────────────────────────────────────────────────────────
from .config import (
    load_config,
    load_system_config,
    save_config,
    SystemConfig,
    SamplingConfig,
    NumericalConfig,
    SelectionObjectiveConfig,
    PipelineConfig,
    VisualizationConfig,
    TwoBodyConfig,
    FullConfig,
)

# ── Pipeline + reporting ─────────────────────────────────────────────
from .pipeline import run_pipeline
from .output.report import generate_run_report
from .output.compare_runs import compare_runs

__all__ = [
    # Constants
    'G_KM', 'M_SUN', 'M_JUP', 'R_JUP', 'R_SUN', 'AU_KM',
    'mu_star', 'mu_planet', 'au_to_km',
    # Dynamics
    'init_hot_jupiter_barycentric', 'restricted_3body_ode', 'simulate_3body',
    # Analysis
    'analyze_trajectory', 'extract_encounter_states', 'EncounterGeometry',
    # Sampling
    'sample_satellite_state_barycentric', 'sample_satellite_state_near_planet',
    # Monte Carlo
    'evaluate_particle', 'run_monte_carlo', 'select_top_indices',
    'select_pareto_indices', 'select_weighted_indices', 'resolve_metric_array',
    # Baselines
    'two_body_hyperbola_from_state', 'monopole_ode',
    'simulate_monopole_baseline', 'compare_3body_with_baselines',
    # Plotting — 3-body diagnostics
    'plot_best_candidate_with_bodies', 'plot_mc_summary', 'plot_velocity_phase_space',
    'plot_star_proximity_distribution', 'plot_planet_frame_diagnostics',
    'plot_multi_candidate_overlay', 'plot_rejection_breakdown',
    'plot_parameter_correlations', 'plot_energy_cdf',
    # Plotting — 2-body parameter-space
    'plot_poincare_heatmaps',
    'plot_oberth_comparison', 'plot_trajectory_tracks',
    # Animation
    'animate_trajectory', 'animate_phase_space', 'generate_all_animations',
    # Config
    'load_config', 'load_system_config', 'save_config',
    'SystemConfig', 'SamplingConfig', 'NumericalConfig', 'SelectionObjectiveConfig', 'PipelineConfig',
    'VisualizationConfig', 'TwoBodyConfig', 'FullConfig',
    # Two-body
    'TwoBodyEncounter', 'TwoBodyGeometry', 'TrajectoryResult',
    'create_encounter_from_config', 'create_planet_encounter_from_config',
    # Comparison
    'compare_2body_3body', 'format_energy', 'print_comparison',
    # Narrowed baselines
    'extract_envelope', 'compute_narrowed_baselines', 'run_narrowed_sweep',
    'EnvelopeParams', 'NarrowedBaselineResult',
    # Pipeline
    'run_pipeline', 'generate_run_report', 'compare_runs',
]
