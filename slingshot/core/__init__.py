"""
Core physics engines and sampling for slingshot-solver.

Submodules:
    twobody_scatter — Ground-truth closed-form 2-body hyperbolic scattering (pure math)
    dynamics        — Restricted 3-body ODE integration
    twobody         — OOP wrapper for TwoBodyScatter parameter-space scans
    sampling        — Initial condition generators (barycentric & planet-frame)
"""

# Ground-truth physics engine
from .twobody_scatter import (
    gravity_assist_no_burn,
    gravity_assist_oberth,
    gravity_assist_closed_form,
    deltaV_lab,
    NoBurnResult,
    OberthResult,
    ScatteringResult,
    _star_velocity_components,
)

# 3-body dynamics
from .dynamics import (
    init_hot_jupiter_barycentric,
    restricted_3body_ode,
    simulate_3body,
)

# 2-body encounter wrapper
from .twobody import (
    TwoBodyEncounter,
    TwoBodyGeometry,
    TrajectoryResult,
    create_encounter_from_config,
    create_planet_encounter_from_config,
)

# Sampling
from .sampling import (
    sample_satellite_state_barycentric,
    sample_satellite_state_near_planet,
)
