"""
Slingshot Solver: Gravitational slingshot dynamics in restricted 3-body systems
"""

__version__ = "2.4.0"

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

from .dynamics import (
    init_hot_jupiter_barycentric,
    restricted_3body_ode,
    simulate_3body,
)

from .analysis import (
    analyze_trajectory,
    extract_encounter_states,
    EncounterGeometry,
)

from .sampling import (
    sample_satellite_state_barycentric,
    sample_satellite_state_near_planet,
)

from .monte_carlo import (
    evaluate_particle,
    run_monte_carlo,
    select_top_indices,
)

from .baselines import (
    two_body_hyperbola_from_state,
    monopole_ode,
    simulate_monopole_baseline,
    compare_3body_with_baselines,
)

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
)

from .animation import (
    animate_trajectory,
    animate_phase_space,
    generate_all_animations,
)

from .config import (
    load_config,
    load_system_config,
    save_config,
    SystemConfig,
    SamplingConfig,
    NumericalConfig,
    PipelineConfig,
    VisualizationConfig,
    TwoBodyConfig,
    FullConfig,
)

from .twobody import (
    TwoBodyEncounter,
    TwoBodyGeometry,
    TrajectoryResult,
    create_encounter_from_config,
    create_planet_encounter_from_config,
)

from .comparison import (
    compare_2body_3body,
    format_energy,
    print_comparison,
)

from .narrowed_baselines import (
    extract_envelope,
    compute_narrowed_baselines,
    run_narrowed_sweep,
    EnvelopeParams,
    NarrowedBaselineResult,
)

from .pipeline import run_pipeline
from .report import generate_run_report
from .compare_runs import compare_runs

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
    # Baselines
    'two_body_hyperbola_from_state', 'monopole_ode',
    'simulate_monopole_baseline', 'compare_3body_with_baselines',
    # Plotting — 3-body diagnostics
    'plot_best_candidate_with_bodies', 'plot_mc_summary', 'plot_velocity_phase_space',
    'plot_star_proximity_distribution', 'plot_planet_frame_diagnostics',
    'plot_multi_candidate_overlay', 'plot_rejection_breakdown',
    'plot_parameter_correlations', 'plot_energy_cdf',
    # Plotting — 2-body parameter-space
    'plot_poincare_heatmaps', 'plot_scattering_maps',
    'plot_encounter_2d_cartesian', 'plot_encounter_2d_trajectories',
    'plot_oberth_comparison',
    # Animation
    'animate_trajectory', 'animate_phase_space', 'generate_all_animations',
    # Config
    'load_config', 'load_system_config', 'save_config',
    'SystemConfig', 'SamplingConfig', 'NumericalConfig', 'PipelineConfig',
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
