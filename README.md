# Slingshot Solver v2.0

**Modular, production-ready Python package for studying gravitational slingshot (gravity assist) dynamics in restricted 3-body systems.**

Star + Hot Jupiter + Satellite orbital mechanics with Monte Carlo simulations, configurable analysis, and video animation.

---

## ✨ Quick Start

### Install
\\\ash
cd slingshot-solver
pip install -e .
\\\

### Run Example
\\\python
import slingshot

# Load config
cfg = slingshot.load_config('config_default.yaml')

# Run Monte Carlo
mc = slingshot.run_monte_carlo(
    N=3000,
    frame="barycentric",
    m_star=1.19 * 1.98847e30,  # kg
    m_p=5.2 * 1.898e27,         # kg
    ...
)

# Select top candidates
top_idx = slingshot.select_top_indices(mc, top_frac=0.10)

# Visualize
slingshot.plot_mc_summary(mc)
slingshot.generate_all_animations(mc['results'][top_idx[0]]['sol'])
\\\

### Or Use Jupyter Notebook
\\\ash
jupyter notebook ThreeBodySolver_v2.ipynb
\\\

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **CHANGELOG.md** | v2.0 implementation details, design decisions, and reference |
| **config_default.yaml** | Configuration template (editable) |
| **ThreeBodySolver_v2.ipynb** | Interactive example notebook |
| **ThreeBodySolver3.ipynb** | Original notebook (v1 reference) |

---

## 🎯 Key Features

🔧 **Modular Architecture** - 9 focused Python modules (dynamics, analysis, sampling, MC, visualization, animation)

⚙️ **Configuration System** - YAML/JSON with Pydantic validation (no hardcoded parameters)

🎬 **Video Animation** - Trajectory + phase-space animations (MP4/GIF)

⚡ **Parallelization** - ProcessPoolExecutor for 3-4x speedup on MC runs

📊 **Robust Analysis** - Encounter geometry extraction with error diagnostics

🔬 **Baseline Comparisons** - 2-body hyperbola & monopole approximation models

📈 **Production-Ready** - Full docstrings, type hints, comprehensive error handling

---

## 📦 Package Structure

\\\
slingshot/
├── __init__.py              # Public API
├── config.py                # Configuration (Pydantic models + YAML loading)
├── dynamics.py              # ODE solver & integration
├── analysis.py              # Trajectory analysis & encounter extraction
├── sampling.py              # Initial condition sampling
├── monte_carlo.py           # MC orchestration & parallelization
├── baselines.py             # Baseline models (2-body, monopole)
├── plotting.py              # Static visualization
└── animation.py             # Video rendering

config_default.yaml          # Configuration template
ThreeBodySolver_v2.ipynb     # Orchestration notebook
requirements.txt             # Dependencies
\\\

---

## 🚀 Core Workflows

### 1. Standard Pipeline
\\\python
from slingshot import *

cfg = load_config('config.yaml')
mc = run_monte_carlo(N=3000, ...)          # Phase 1: Monte Carlo
top = select_top_indices(mc, top_frac=0.1) # Phase 2: Selection
plot_mc_summary(mc)                        # Phase 3: Visualization
\\\

### 2. Parameter Sweep
\\\python
for n_particles in [1000, 3000, 5000]:
    cfg.pipeline.N_particles = n_particles
    save_config(cfg, f'run_{n_particles}.yaml')
    # Run experiments programmatically
\\\

### 3. Custom Analysis
\\\python
from slingshot.dynamics import simulate_3body
from slingshot.analysis import analyze_trajectory

sol = simulate_3body(Y0, t_span, m_star, m_p)
ana = analyze_trajectory(sol, frame="barycentric")
print(f"Δv = {ana['delta_v']:.2f} km/s")
\\\

---

## 📋 Physics Model

**Restricted Planar 3-Body Problem (PCR3BP)**

System: Star + Hot Jupiter + Satellite  
Integration: High-precision ODE (RK45, rtol=1e-10, atol=1e-10)  
Frames: Barycentric inertial & planet-relative

**Key Metrics:**
- Velocity change (Δv): Energy gain from slingshot encounter
- Deflection angle: Angular deviation of exit velocity
- Specific energy: Determines escape vs. bound trajectories
- Impact parameter: Encounter strength

---

## 🔧 Configuration

Edit \config_default.yaml\ or create custom configs:

\\\yaml
system:
  name: Kepler-432              # System identifier
  M_star_Msun: 1.19            # Star mass
  M_planet_Mjup: 5.2           # Planet mass
  a_planet_AU: 0.0896          # Orbital semi-major axis

sampling:
  mode: barycentric            # or "planet"
  v_mag_min_kms: 10.0          # Velocity range
  v_mag_max_kms: 120.0
  impact_param_min_AU: 0.5     # Impact parameter range
  impact_param_max_AU: 3.0

pipeline:
  N_particles: 3000            # Monte Carlo size
  t_mc_max_sec: 1.0e7          # Simulation duration
  select_metric: bary_delta_v_pct
  n_parallel: null             # null=serial, N=parallel workers

visualization:
  render_video: true
  animate_trajectory: true
  animate_phase_space: true
\\\

---

## 🎬 Animations

Automatically generates videos for each test case:

- **Type A** (Trajectory): Star/planet/satellite paths with trailing history
- **Type B** (Phase-space): Velocity evolution (v_radial vs v_normal)
- **Type C** (Comparison): Framework for multi-trajectory overlays (extensible)

\\\python
from slingshot.animation import generate_all_animations

animations = generate_all_animations(
    sol,
    output_dir="./frames",
    video_fps=30,
    video_format="mp4"  # or "gif"
)
\\\

---

## ⚡ Performance

**Typical Runtime** (N=3000 particles, Kepler-432):

| Mode | Duration | Notes |
|------|----------|-------|
| Serial | 2-3 hours | Single-threaded |
| Parallel (4 cores) | 40-50 min | 3-4x speedup |

Enable parallelization in config:
\\\yaml
pipeline:
  n_parallel: 4  # Use 4 worker processes
\\\

---

## 📊 Key Improvements from v1

| Feature | v1 | v2 |
|---------|----|----|
| Code Structure | Monolithic (1,200 lines) | Modular (9 modules, 2,250 LOC) |
| Configuration | Hardcoded | External YAML/JSON |
| Duplication | ~400 lines | 0 lines |
| Parallelization | ❌ | ✅ ProcessPoolExecutor |
| Animation | ❌ | ✅ Type A+B + framework |
| Robustness | ⚠️ | ✅ EncounterGeometry class |
| Error Diagnostics | Minimal | Detailed reason codes |
| Reproducibility | Manual | Config files |
| Testability | Difficult | Modular functions |

---

## 🛠️ API Reference

### Core Functions

**Dynamics**
\\\python
simulate_3body(Y0, t_span, m_star, m_p, n_eval, rtol, atol)
\\\

**Analysis**
\\\python
analyze_trajectory(sol, frame="barycentric"|"planet", ...)
extract_encounter_states(sol, m_p, R_p, ...)
\\\

**Sampling**
\\\python
sample_satellite_state_barycentric(Y_sp0, N, v_mag_min, ...)
sample_satellite_state_near_planet(Y_sp0, N, r_min, ...)
\\\

**Monte Carlo**
\\\python
run_monte_carlo(N, t_span, m_star, m_p, frame, ..., n_parallel=None)
select_top_indices(mc, metric, sign, top_frac)
\\\

**Visualization**
\\\python
plot_mc_summary(mc)
plot_best_candidate_with_bodies(sol, analysis)
generate_all_animations(sol, output_dir, ...)
\\\

**Configuration**
\\\python
load_config(path)
save_config(cfg, path)
load_system_config(system_name)  # Kepler-432, TOI-1431, etc.
\\\

See module docstrings for full parameter documentation.

---

## 🔍 Troubleshooting

**ImportError: No module named 'slingshot'**
\\\ash
pip install -e .
\\\

**ValidationError loading config**
- Check YAML syntax (spaces, types)
- Verify all required fields present
- See \config.py\ for schema

**Parallelization not working**
- Ensure \
_parallel > 1\ in config
- Check disk space for temp files
- Try \
_parallel=None\ (serial mode)

**Animation generation fails**
- Install ffmpeg: \fmpeg -version\
- Check disk space
- Try GIF format: \ideo_format: gif\

---

## 🔗 Related Files

- **Original Project**: \ThreeBodySolver3.ipynb\ (v1 reference)
- **System Presets**: \config_default.yaml\
- **Dependencies**: \equirements.txt\
- **Notebooks**: \ThreeBodySolver_v2.ipynb\ (orchestration example)

---

## 📖 For More Information

See **CHANGELOG.md** for:
- Detailed implementation summary
- Design decisions and rationale
- Extensibility guide
- Architecture and API details
- Complete documentation map

---

## 🎓 Learning Path

1. **Install & setup** (5 min): Run installation & load config
2. **Run example** (15 min): Execute \ThreeBodySolver_v2.ipynb\
3. **Customize** (10 min): Modify config, change parameters
4. **Explore** (30+ min): Examine modules, write custom analyses
5. **Extend** (optional): Add new analysis/animation types

---

## 📝 Requirements

\\\
numpy
scipy
matplotlib
pyyaml
pydantic
\\\

Optional:
- ffmpeg-python (for advanced video features)
- pytest (development/testing)

---

## 📄 License

[Specify your license here]

---

**Version**: 2.0.0  
**Status**: Production-Ready ✅  
**Last Updated**: February 3, 2026

**Start here**: Read [CHANGELOG.md](CHANGELOG.md) or open [ThreeBodySolver_v2.ipynb](ThreeBodySolver_v2.ipynb)
