# ThreeBodySolver3: Restricted 3-Body Slingshot Dynamics Simulator

## Overview

**ThreeBodySolver3.ipynb** is a comprehensive numerical simulator for studying gravitational slingshot maneuvers in a **restricted planar 3-body system**. It models a star, a hot Jupiter exoplanet, and a massless test particle (satellite/comet) to analyze how the satellite can gain or lose energy through gravitational interaction with the planetary system.

The code implements high-precision numerical integration, Monte Carlo parameter sweeps, and multi-frame analysis to characterize slingshot trajectories for potential spaceflight mission design or astrophysical research.

---

## Physics & Science

### The Restricted Planar 3-Body Problem

The simulator solves the **planar circular restricted 3-body problem (PCR3BP)** with three bodies:

1. **Star** (mass $M_{\star}$): Central massive body
2. **Hot Jupiter** (mass $M_p$): Orbiting the star at semi-major axis $a_p$
3. **Satellite** (mass $\approx 0$): Test particle that experiences gravitational forces but does not back-react on the star or planet

#### Governing Equations

The equations of motion in the **barycentric inertial frame** are:

$$\ddot{\mathbf{r}}_s = -\frac{GM_p}{|\mathbf{r}_s - \mathbf{r}_p|^3}(\mathbf{r}_s - \mathbf{r}_p) - \frac{GM_{\star}}{|\mathbf{r}_s - \mathbf{r}_{\star}|^3}(\mathbf{r}_s - \mathbf{r}_{\star})$$

where:
- $\mathbf{r}_s$, $\mathbf{r}_p$, $\mathbf{r}_{\star}$ are barycentric position vectors
- $G$ is the gravitational constant (in km³/(kg·s²))
- The star and planet orbit their common barycenter in a quasi-circular orbit

### Physical Constants Used

The code uses SI-consistent units with distances in **kilometers** and time in **seconds**:

- $G = 6.67430 \times 10^{-20}$ km³/(kg·s²)
- $M_{\odot} = 1.98847 \times 10^{30}$ kg (Solar mass)
- $M_{Jup} = 1.898 \times 10^{27}$ kg (Jupiter mass)
- $R_{Jup} = 71,492$ km (Jupiter radius)
- 1 AU = $1.495978707 \times 10^8$ km

### System Configurations

The simulator includes built-in configurations for real exoplanetary systems. Default system: **Kepler-432**

- **Kepler-432b** (primary configuration):
  - Star: 1.19 $M_{\odot}$ (K0V spectral type)
  - Planet: 5.2 $M_{Jup}$ hot Jupiter
  - Orbital period: ~52.5 days
  - Semi-major axis: 0.0896 AU (~13.4 million km)

Alternative: TOI-1431 (commented in code)

### Slingshot Mechanics

A **slingshot maneuver** (or gravity assist) occurs when a satellite approaches a gravitational body, gets deflected, and exits with a different velocity. In the 3-body context:

- **Planet-frame analysis**: Measures velocity change relative to the hot Jupiter ($\Delta v_{planet}$)
- **Barycentric-frame analysis**: Measures velocity change in the inertial reference frame ($\Delta v_{barycentric}$)
- **Deflection angle**: The angular deviation of the exit velocity from entry velocity
- **Impact parameter**: Closest perpendicular approach distance to the system's barycenter
- **Specific energy**: Determines if the satellite escapes ($\epsilon > 0$) or remains bound ($\epsilon < 0$)

---

## Code Structure & Features

### 1. **Initialization & Configuration**

**Cell 1: Setup**
- Imports necessary libraries (NumPy, SciPy, Matplotlib)
- Defines global constants (G, masses, radii)
- **System selection**: Choose between Kepler-432, TOI-1431, or custom systems
- Sets numerical tolerances (RTOL=1e-10, ATOL=1e-10) for high-precision integration

**Key Functions:**
- `init_hot_jupiter_barycentric()`: Computes initial barycentric state for the star-planet system at a given orbital phase

### 2. **Numerical Integration**

**Cell 2-4: ODE Solver**

The code uses **scipy.integrate.solve_ivp** with dense output to integrate 12-DOF system:

$$Y = [x_{\star}, y_{\star}, \dot{x}_{\star}, \dot{y}_{\star}, x_p, y_p, \dot{x}_p, \dot{y}_p, x_s, y_s, \dot{x}_s, \dot{y}_s]$$

**Key Function:**
- `restricted_3body_ode()`: Computes accelerations for all three bodies
- `simulate_3body()`: Wrapper integrating over specified time span with configurable output density

**Features:**
- High relative/absolute tolerance (1e-10) for accuracy
- Dense output for trajectory reconstruction
- Error handling and validation

### 3. **Trajectory Analysis**

**Cells 2-3: Analysis Functions**

**Single Trajectory Analysis:**
- `analyze_3body_hyperbolic_barycentric()`: Extracts barycentric-frame metrics
  - Initial/final barycentric speeds ($v_i$, $v_f$)
  - Energy gain ($\Delta v = v_f - v_i$)
  - Deflection angle
  - Specific orbital energy ($\varepsilon$, determines if unbound)
  - Closest approach distance
  
- `analyze_3body_slingshot()`: Extracts planet-frame metrics
  - Relative velocity changes in the planet frame
  - Planet-relative deflection angle
  - 2-body orbital parameters equivalent

**Encounter State Extraction:**
- `extract_encounter_states()`: Identifies three key trajectory phases:
  - **"Far in"**: Initial state when satellite is far from planet
  - **Periapsis**: Closest approach to planet
  - **"Far out"**: Final state when satellite has escaped

### 4. **Initial Condition Sampling**

**Cells 3-4: Parametric Sampling**

Two complementary sampling modes:

**A. Barycentric-Frame Sampling** (recommended for hyperbolic encounters):
- `sample_satellite_state_barycentric()`: 
  - Control incoming velocity magnitude: $v_i \in [10, 120]$ km/s
  - Impact parameter: $b \in [0.5 \times a_p, 3 \times a_p]$ km
  - Incoming angle (radial vs tangential): $\theta \in [-60°, 60°]$
  - Azimuthal entry angle: $\phi \in [0°, 360°]$ (isotropic)

**B. Planet-Relative Sampling** (for orbiting satellites):
- `sample_satellite_state_near_planet()`:
  - Orbital radius: $r \in [20 R_p, 5 \times a_p / R_p]$ 
  - Relative velocity: $v_{rel} \in [12, 80]$ km/s
  - Random orbital position and azimuthal direction

### 5. **Monte Carlo Simulation**

**Cell 5: Batch Monte Carlo**

`run_batch_mc_3body_barycentric()`: Execute N test particles with:
- Random sampling of initial conditions
- Integrated 3-body trajectories
- Automatic rejection of non-physical cases (collisions, failures)
- Parallel analysis across all trajectories
- Returns success indicators and metrics arrays

**Default Configuration:**
- N = 3,000 test particles
- Simulation duration: $t_{max} = 10^7$ seconds (~115 days)
- Success rate tracking

**Output Metrics (per particle):**
- `ok`: Boolean success indicator
- `delta_v_bary`: Barycentric velocity change
- `deflection_bary`: Barycentric deflection angle
- `r_min`: Minimum distance to planet
- `eps_i_bary`, `eps_f_bary`: Specific energies
- `unbound_bary`: Whether final state is unbound

### 6. **Top Candidate Selection**

**Cell 5: Selection Logic**

`select_top_indices_by_delta_v_with_bary()`: Rank and filter trajectories by:
- **Metrics**: 
  - Planet-frame $\Delta v$ (default)
  - Barycentric $\Delta v$
  - Percent velocity change: $\Delta v / v_i \times 100\%$
- **Unbound Requirements**:
  - None: Accept any trajectory
  - 'pre': Initial state must be unbound
  - 'post': Final state must be unbound
  - 'either'/'both': Logical combinations
- **Selection**: Top N% by chosen metric (default 10% of successful cases)

### 7. **High-Resolution Re-run**

**Cell 5: Refinement**

`rerun_top_candidates_3body()`: For selected trajectories:
- Re-integrate at longer time span ($t_{max} = 10^7$ s)
- Evaluate with finer output resolution (e.g., 1,000 points vs. default)
- Compute detailed analysis for best candidates

### 8. **Visualization & Diagnostics**

**Cells 6-10: Plotting Functions**

**A. Summary Statistics:**
- `plot_mc_summary()`: Scatter plot of deflection vs. Δv, histogram of deflection fractions
- `summary_statistics_for_sol()`: Print energy, velocity, and unbound indicators

**B. Trajectory Plots:**
- `plot_best_candidate_with_bodies()`: Barycentric frame with scaled circles for star and planet
  - Trajectories of all three bodies
  - Physical sizes of star and planet rendered
  - Analysis text box with key metrics

**C. Phase-Space Analysis:**
- `plot_velocity_phase_space()`: Barycentric and planet-relative velocity components
  - $v_x$ vs. $v_y$ (Cartesian velocity space)
  - $v_{radial}$ vs. $v_{tangential}$ (orbital velocity components)

**D. Encounter Geometry:**
- `extract_encounter_states()`: Identify and extract asymptotic regimes
- `two_body_hyperbola_from_state()`: Compute equivalent 2-body hyperbolic orbit parameters
- `sample_hyperbola_orbit()`: Generate analytical hyperbola near periapsis
- `hyperbola_to_planet_frame()`: Rotate hyperbola into planet-frame coordinates

### 9. **Comparison with Baseline Models**

**Cells 9-11: Comparative Analysis**

`compare_3body_with_baselines()`: For best trajectory, compute:

**2-Body Planet-Only Baseline:**
- Treat satellite as test particle around planet alone
- Extract equivalent hyperbolic orbital elements
- Compute 2-body deflection angle
- Compare 3-body vs. pure 2-body outcome

**Monopole Barycentric Baseline:**
- Treat combined star+planet as single point mass at barycenter
- Integrate same trajectory under monopole potential
- Compare energy and angular momentum changes
- Quantify extra effects from planet's presence

**Outputs:**
- Deflection angle differences (2-body vs. 3-body)
- Energy/angular momentum changes ($\Delta \varepsilon$, $\Delta h$)
- Visualization overlays comparing trajectories in both frames

---

## Usage & Workflow

### Quick Start

1. **Open ThreeBodySolver3.ipynb** in Jupyter
2. **Cell 1**: Execute setup and configure system parameters
3. **Cell 5**: Adjust Monte Carlo parameters (N_MC, sampling mode, selection metric)
4. **Cell 5**: Run `run_full_pipeline_hot_jupiter()`
   - Completes all 4 phases: MC → Selection → Re-run → Visualization
5. **Cells 6-11**: Examine best candidate in detail
   - Phase-space plots
   - Baseline comparisons
   - Detailed statistics

### Configuration Parameters

All parameters are user-adjustable in **Cell 5** (before running pipeline):

**Monte Carlo Phase:**
```python
N_MC = 3000                      # Number of test particles
T_MC_MAX = 1e7                   # Simulation duration (seconds)
SAMPLING_MODE = "barycentric"   # "planet" or "barycentric"
```

**Barycentric Sampling (default):**
```python
SAT_V_MAG_MIN_RUN = 10          # km/s
SAT_V_MAG_MAX_RUN = 120         # km/s
SAT_IMPACT_PARAM_MIN_RUN = 0.5 * A_PLANET_KM
SAT_IMPACT_PARAM_MAX_RUN = 3.0 * A_PLANET_KM
SAT_ANGLE_IN_MIN_RUN = -60.0    # degrees
SAT_ANGLE_IN_MAX_RUN = 60.0     # degrees
```

**Selection Criteria:**
```python
TOP_FRAC = 0.10                 # Top 10% of successful cases
SELECT_METRIC = 'bary_delta_v_pct'  # Optimization metric
SELECT_SIGN = 'maximize'        # Maximize or minimize
BARY_UNBOUND_REQUIREMENT = 'both'   # Escape requirement
```

---

## Key Functions Reference

| Function | Purpose |
|----------|---------|
| `init_hot_jupiter_barycentric()` | Initialize star+planet barycentric state |
| `restricted_3body_ode()` | ODE for 3-body accelerations |
| `simulate_3body()` | Integrate 3-body trajectory |
| `analyze_3body_hyperbolic_barycentric()` | Extract barycentric metrics |
| `analyze_3body_slingshot()` | Extract planet-frame metrics |
| `sample_satellite_state_barycentric()` | Generate random initial conditions |
| `run_batch_mc_3body_barycentric()` | Execute full Monte Carlo |
| `rerun_top_candidates_3body()` | High-resolution re-run |
| `plot_mc_summary()` | Summary scatter and histograms |
| `plot_best_candidate_with_bodies()` | Detailed trajectory plot |
| `compare_3body_with_baselines()` | Compare against 2-body and monopole models |

---

## Physical Interpretation of Key Outputs

### Velocity Metrics

- **$\Delta v$ (planet frame)**: Energy gain relative to the moving planet
  - Positive: Satellite accelerated by encounter
  - Negative: Satellite decelerated
  - Typical range: -50 to +50 km/s for hot Jupiters

- **$\Delta v$ (barycentric)**: Energy gain in inertial frame
  - Smaller than planet-frame values (planet is also moving)
  - Determines if satellite escapes the system

### Energy States

- **$\varepsilon = \frac{1}{2}v^2 - \frac{GM}{r}$ (specific orbital energy)**:
  - $\varepsilon > 0$: Unbound (escapes to infinity)
  - $\varepsilon = 0$: Marginal (parabolic trajectory)
  - $\varepsilon < 0$: Bound (remains in system)

### Trajectory Geometry

- **Deflection angle**: Indicates how much the planet bent the satellite's path
  - Near 180°: Nearly complete reversal (strong encounter)
  - Near 0°: Weak encounter (large miss distance)

- **Impact parameter**: Controls encounter strength
  - Small $b$: Satellite passes close, experiences large deflection
  - Large $b$: Satellite passes far, experiences weak deflection

---

## Notes on Integration Strategy

The simulator uses **scipy.integrate.solve_ivp** with high tolerances:
- **RTOL = 1e-10**: Relative tolerance on state variables
- **ATOL = 1e-10**: Absolute tolerance on state variables

This ensures:
- Accurate energy conservation over long simulations (100+ days)
- Reliable detection of closest approach
- Valid identification of asymptotic regimes

For production use, further optimizations could include:
- Symplectic integrators (Leapfrog, Störmer-Verlet) for long-term stability
- Adaptive step-size control based on local curvature
- Event detection for periapsis/apoapsis

---

## Related Files in Directory

- **SimulationBackEnd.py**: Alternative 2-body leapfrog integrator with perigee detection
- **simulation_functions.py**: Legacy integration functions (2-body Kepler orbit solver)
- **ThreeBodySolver.ipynb, ThreeBodySolver2.ipynb**: Earlier iterations (less complete)
- **TwoBodySolver*.ipynb**: Restricted 2-body solvers for reference
- **MCMC2Body.ipynb**: Bayesian parameter inference for 2-body orbits
- **NewSolver/ODESolver.ipynb**: Circular restricted 3-body problem solver (alternative)
- **frames/**: Output animation frames for slingshot trajectories
- **Output files**: `three_body_simulation.csv`, `three_body_leapfrog.csv`, animation videos

---

## Future Improvements

1. **Escape/impact detection**: Add event handling to stop integration if satellite escapes system
2. **Non-planar 3D orbits**: Extend to full 3D restricted 3-body problem
3. **Resonances**: Search for periodic/quasi-periodic orbits (Lyapunov, halo orbits)
4. **Perturbation analysis**: Compute stable/unstable manifolds
5. **Real mission design**: Parameterize for actual spacecraft trajectories
6. **Ensemble uncertainty**: Propagate uncertainties in stellar/planetary parameters

---

## References & Theory

### Key Concepts
- Restricted planar circular 3-body problem (PCR3BP)
- Gravity assist / slingshot maneuvers
- Hyperbolic orbits and deflection angles
- Orbital energy and specific angular momentum

### Computational Methods
- Dense output ODE integration (Runge-Kutta 4/5)
- Adaptive step-size control
- Monte Carlo parameter sweeps
- Frame transformations (barycentric ↔ planet-relative)

### Exoplanet Systems
- Hot Jupiters: Short-period, massive planets orbiting close to host stars
- Kepler-432 system: Well-characterized system with precise orbital measurements
- System parameters from NASA Exoplanet Archive

---

**Version**: ThreeBodySolver3.ipynb (Current)  
**Last Updated**: February 2026  
**Status**: Research/Educational Use

