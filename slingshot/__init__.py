"""
Slingshot Solver: Gravitational slingshot dynamics in restricted 3-body systems
"""

__version__ = "2.0.0"

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
)

from .animation import (
    animate_trajectory,
    animate_phase_space,
    generate_all_animations,
)

from .config import (
    load_config,
    load_system_config,
    SystemConfig,
    SamplingConfig,
    PipelineConfig,
)

__all__ = [
    'init_hot_jupiter_barycentric',
    'restricted_3body_ode',
    'simulate_3body',
    'analyze_trajectory',
    'extract_encounter_states',
    'EncounterGeometry',
    'sample_satellite_state_barycentric',
    'sample_satellite_state_near_planet',
    'evaluate_particle',
    'run_monte_carlo',
    'two_body_hyperbola_from_state',
    'monopole_ode',
    'simulate_monopole_baseline',
    'compare_3body_with_baselines',
    'plot_best_candidate_with_bodies',
    'plot_mc_summary',
    'plot_velocity_phase_space',
    'animate_trajectory',
    'animate_phase_space',
    'generate_all_animations',
    'load_config',
    'load_system_config',
    'SystemConfig',
    'SamplingConfig',
    'PipelineConfig',
    'VisualizationConfig',
]
