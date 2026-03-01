"""
Analysis layer for slingshot-solver.

Submodules:
    trajectory         — Trajectory analysis and encounter geometry extraction
    monte_carlo        — Monte Carlo batch processing, selection, filtering
    baselines          — 2-body hyperbola + monopole baseline models
    narrowed_baselines — Envelope-matched 2-body baseline sweeps
    comparison         — 2-body vs 3-body energy comparison utilities
"""

from .trajectory import (
    analyze_trajectory,
    extract_encounter_states,
    EncounterGeometry,
)

from .monte_carlo import (
    evaluate_particle,
    run_monte_carlo,
    select_top_indices,
    select_pareto_indices,
    select_weighted_indices,
    resolve_metric_array,
)

from .baselines import (
    two_body_hyperbola_from_state,
    monopole_ode,
    simulate_monopole_baseline,
    compare_3body_with_baselines,
)

from .narrowed_baselines import (
    extract_envelope,
    compute_narrowed_baselines,
    run_narrowed_sweep,
    EnvelopeParams,
    NarrowedBaselineResult,
)

from .comparison import (
    compare_2body_3body,
    format_energy,
    print_comparison,
)
