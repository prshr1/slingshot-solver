# Slingshot Solver v3.0

**Production-grade Python package for studying gravitational slingshot (gravity-assist) dynamics in restricted 3-body systems, with a focus on interstellar-velocity encounters.**

Layered architecture with **Core** (dynamics, 2-body math), **Analysis** (Monte Carlo, trajectory metrics), and **Output** (plotting, animation, reporting) layers. Star + Hot Jupiter + Satellite orbital mechanics with Monte Carlo simulations, 2-body baselines, star-proximity filtering, planet-frame diagnostics, auto-generated reports, video animation, and full test suite. Installable package with CLI entry point.

**Canonical unit system**: km-kg-s throughout. Energies in km²/s² (≡ MJ/kg).

---

## Quick Start

### Install

```bash
cd slingshot-solver
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

### CLI Entry Point

After install, use the `slingshot` command:

```bash
slingshot configs/config_kepler432_case.yaml
slingshot compare results/run_a results/run_b
slingshot --help
```

Or call directly via `python run.py`.

This runs the full 8-phase pipeline and produces a timestamped results directory:

```
results/results_Kepler-432_YYYYMMDD_HHMMSS/
├── config.yaml          # Frozen copy of the config used
├── summary.csv          # All candidates ranked by ΔV
├── results.pkl          # Full MC data + rerun solutions (pickle)
├── REPORT.md            # Auto-generated analysis report
├── *.png                # All diagnostic plots
└── frames/              # Animation frames (if enabled)
```

#### CLI Options

```bash
python run.py configs/config.yaml --output-dir results/my_run   # custom output dir
python run.py configs/config.yaml --skip-plots                   # data only, no figures
python run.py configs/config.yaml --skip-animations              # skip video render
python run.py configs/config.yaml --phases mc,select,rerun       # run only specific phases
python run.py configs/config.yaml --quiet                        # minimal console output
```

### Compare Runs

```bash
python run.py compare results/run_a results/run_b results/run_c
```

Prints a side-by-side comparison table of system parameters, particle counts, and best ΔV from each run.

### Run Individual Phases (Python)

Each pipeline phase is independently callable for debugging:

```python
from slingshot.config import load_config
from slingshot.pipeline import phase_monte_carlo, phase_select, phase_rerun

cfg = load_config('configs/config_kepler432_case.yaml')
mc = phase_monte_carlo(cfg, verbose=True)
top_idx = phase_select(cfg, mc, verbose=True)
rerun = phase_rerun(cfg, mc, top_idx, verbose=True)
```

### Interactive Notebook

| Notebook | Purpose |
|----------|---------|
| `ThreeBodySolver_v2.ipynb` | 3-body Monte Carlo exploration, analysis, and visualisation |

---

## Architecture

**Layered package structure** (v3.0 restructure):

```
slingshot/
├── __init__.py                     # Top-level re-exports (72 symbols)
├── config.py                       # Pydantic models + YAML loader
├── constants.py                    # G_KM, M_SUN, M_JUP, R_JUP, R_SUN, AU_KM
├── console.py                      # Pretty-print utilities
├── cli.py                          # CLI entry point (argparse)
├── pipeline.py                     # 8-phase orchestrator
│
├── core/                           # Core dynamics & mathematics
│   ├── __init__.py
│   ├── dynamics.py                 # 3-body ODE + RK integration
│   ├── twobody.py                  # TwoBodyEncounter grid-scan class
│   ├── twobody_scatter.py          # Closed-form hyperbolic solver (ground truth)
│   └── sampling.py                 # Initial condition generation
│
├── analysis/                       # Analysis workflows & metrics
│   ├── __init__.py
│   ├── trajectory.py               # Trajectory analysis + EncounterGeometry
│   ├── monte_carlo.py              # Monte Carlo sweep + candidate selection
│   ├── baselines.py                # 2-body hyperbola + monopole baselines
│   ├── narrowed_baselines.py       # Post-hoc narrowed 2-body comparisons
│   └── comparison.py               # 2-body vs 3-body cross-comparison
│
├── output/                         # Output (plots, reports, animation)
│   ├── __init__.py
│   ├── plotting.py                 # 3-body diagnostic plots (9 functions)
│   ├── plotting_twobody.py         # 2-body heatmaps & encounter maps (5 functions)
│   ├── animation.py                # Video rendering (trajectory + phase-space)
│   ├── report.py                   # Auto-generated REPORT.md
│   └── compare_runs.py             # Cross-run comparison tables

Workspace:
├── configs/                        # YAML configuration files (5 presets + custom)
├── results/                        # All outputs (gitignored)
│   └── results_Kepler-432_*/       # Per-run dirs (config, plots, data, REPORT.md)
├── tests/                          # Test suite (29 tests, all passing)
├── pyproject.toml                  # Modern packaging (setuptools, entry point)
├── requirements.txt                # Dependencies
├── run.py                          # Thin wrapper → slingshot.cli:main
├── ThreeBodySolver.ipynb           # Interactive notebook (v3.0 imports)
└── README.md
```

### Data Flow

```
┌───────────────────────────────────────────────────────────┐
│              configs/config_kepler432_case.yaml            │
│         (Single source of truth — Pydantic schema)        │
└─────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
              python run.py config.yaml
                      │
        ┌─────────────┼─────────────────────────┐
        ▼             ▼                         ▼
   Phase 1: MC    Phase 5: Baselines       Phase 6: Plots
   monte_carlo    baselines.py             plotting.py
   sampling.py    narrowed_baselines.py    plotting_twobody.py
        │         comparison.py                 │
        ▼             │                         │
   Phase 2: Select    │                    Phase 7: Animations
   Phase 3: Rerun     │                    animation.py
   Phase 4: Best      │                         │
        │             │                         │
        └─────────────┴─────────────────────────┘
                      │
                      ▼
              Phase 8: Save + Report
              report.py → REPORT.md
              results/results_Kepler-432_YYYYMMDD_HHMMSS/
```

---

## Unit System

All modules use **km-kg-s** consistently:

| Quantity | Unit | Note |
|----------|------|------|
| Distance | km | |
| Velocity | km/s | |
| Mass | kg | |
| Time | s | |
| Energy | km²/s² | ≡ MJ/kg (since 1 km²/s² = 10⁶ J/kg) |
| G | 6.67430 × 10⁻²⁰ km³ kg⁻¹ s⁻² | `slingshot.constants.G_KM` |
| μ | km³/s² | G × M |

Constants are defined **once** in `slingshot/constants.py` and imported by every module.

---

## Configuration

Configs live in `configs/`. Edit or create new ones:

```yaml
system:
  name: Kepler-432
  M_star_Msun: 1.19
  R_star_Rsun: 4.06
  M_planet_Mjup: 5.2
  R_planet_Rjup: 1.155
  a_planet_AU: 0.0896
  bulk_velocity_vx_kms: 0.0
  bulk_velocity_vy_kms: 0.0

sampling:
  mode: barycentric
  v_mag_min_kms: 10.0
  v_mag_max_kms: 120.0
  impact_param_min_AU: 0.5
  impact_param_max_AU: 3.0

numerical:
  rtol: 1.0e-10
  atol: 1.0e-10
  star_min_clearance_Rstar: 1.0   # reject star-penetrating orbits

pipeline:
  N_particles: 3000
  t_mc_max_sec: 1.0e7
  select_mode: single
  select_metric: bary_delta_v_pct
  select_sign: maximize
  selection_objectives:
    - metric: bary_delta_v
      sign: maximize
      weight: 1.0
    - metric: delta_v_vec
      sign: maximize
      weight: 1.0
    - metric: energy_from_planet_orbit
      sign: maximize
      weight: 1.0
  weighted_normalization: minmax
  n_parallel: null
```

Pipeline consistency note: 2-body diagnostic maps generated by `run.py`/`run_pipeline` auto-sync the full star velocity vector from the 3-body run (narrowed-envelope value when available; otherwise initial star barycentric velocity).

Selection modes:
- `single`: one metric (`select_metric`, `select_sign`)
- `pareto`: non-dominated sorting over `selection_objectives`
- `weighted`: normalized weighted score over `selection_objectives`

Load in Python:

```python
from slingshot.config import load_config
cfg = load_config('configs/config_kepler432_case.yaml')
```

---

## Core Workflows

### 1. Full Pipeline (One Command)

```bash
# Installation method 1: editable install
slingshot configs/config_kepler432_case.yaml

# Installation method 2: via Python
python run.py configs/config_kepler432_case.yaml
```

Runs all 8 phases: MC → Select → Rerun → Best → Baselines → Plots → Animations → Save. Produces timestamped `results/results_Kepler-432_YYYYMMDD_HHMMSS/` with `REPORT.md`, diagnostic plots, and pickled data.

### 2. Individual Phases (Python)

```python
from slingshot.config import load_config
from slingshot.pipeline import (
    phase_monte_carlo, phase_select, phase_rerun,
    phase_best_selection, phase_baselines, phase_plots,
)

cfg = load_config('configs/config_kepler432_case.yaml')
mc = phase_monte_carlo(cfg, verbose=True)
top_idx = phase_select(cfg, mc, verbose=True)
rerun = phase_rerun(cfg, mc, top_idx, verbose=True)
best = phase_best_selection(
    rerun["analyses"], top_idx, rerun["solutions"], verbose=True
)
baselines = phase_baselines(cfg, rerun["analyses"], best["best_vec_ana"], verbose=True)
```

### 3. Two-Body Heatmaps

Config-driven 2-body encounter visualisations via the **Analysis** layer:

```python
from slingshot.analysis.narrowed_baselines import compute_narrowed_baselines
from slingshot.output.plotting_twobody import plot_trajectory_tracks

narrowed = compute_narrowed_baselines(
    analyses_top=[...],
    cfg=cfg,
    padding_factor=1.5,
)

figs = plot_trajectory_tracks(
    narrowed=narrowed,
    sols_best=sols,
    analyses_best=analyses,
    cfg=cfg,
)
```

Auto-generated by the pipeline when `visualization.render_2body_tracks: true`.

### 4. Cross-Run Comparison

```python
from slingshot.output.compare_runs import compare_runs

compare_runs(["results/run_a", "results/run_b"])
```

Or CLI:

```bash
slingshot compare results/run_a results/run_b
```

### 5. Animations

```python
from slingshot.output.animation import generate_all_animations

animations = generate_all_animations(
    best_sol,
    output_dir="./results/frames",
    video_fps=30,
    video_format="mp4",
)
```

---

## API Reference

### Top-Level Imports

```python
# All public symbols re-exported from slingshot (72 symbols)
from slingshot import (
    # Constants
    G_KM, M_SUN, M_JUP, R_JUP, R_SUN, AU_KM,
    # Config
    load_config, save_config, FullConfig,
    # Core dynamics
    simulate_3body, init_hot_jupiter_barycentric,
    # Analysis
    analyze_trajectory, run_monte_carlo,
    # Plotting & output
    plot_mc_summary, generate_all_animations,
    # Pipeline orchestration
    run_pipeline,
    # + 50+ more...
)
```

See `slingshot/__init__.py` for the complete list.

### Constants (`slingshot.constants`)

| Symbol | Value | Description |
|--------|-------|-------------|
| `G_KM` | 6.67430e-20 | Gravitational constant [km³ kg⁻¹ s⁻²] |
| `M_SUN` | 1.98847e30 | Solar mass [kg] |
| `M_JUP` | 1.898e27 | Jupiter mass [kg] |
| `R_JUP` | 71492.0 | Jupiter radius [km] |
| `R_SUN` | 696000.0 | Solar radius [km] |
| `AU_KM` | 1.495978707e8 | Astronomical unit [km] |

Helpers: `mu_star(M_Msun)`, `mu_planet(M_Mjup)`, `au_to_km(au)`, `km_to_au(km)`

### Core Layer (`slingshot.core`)

**Dynamics** (`slingshot.core.dynamics`):
- `simulate_3body(Y0, t_span, m_star, m_p, n_eval, rtol, atol)` — RK45 integration with escape handling
- `init_hot_jupiter_barycentric(a_km, m_star, m_p, phase, prograde)` — barycentric IC generation

**Two-Body** (`slingshot.core.twobody`):
- `TwoBodyEncounter(M_body_kg, G, label)` — encounter manager
- `TwoBodyEncounter.scan_parameter_space(...)` — grid scan over b × angle
- `TwoBodyEncounter.get_energy_statistics(energies)` — distribution stats

**Two-Body Scatter** (`slingshot.core.twobody_scatter`):
- `gravity_assist_oberth(v_vec, dv, p_r, mu)` — closed-form hyperbolic solver (ground truth)
- `ga_no_burn(v_mag, b, mu)`, `ga_oberth(v_mag, b, dv, mu)` — utility functions

**Sampling** (`slingshot.core.sampling`):
- `sample_ic_uniform(...)` — uniform parameter space sampling
- `calc_escape_radius(...)`, `calc_min_hill(...)` — geometry helpers

### Analysis Layer (`slingshot.analysis`)

**Trajectory** (`slingshot.analysis.trajectory`):
- `analyze_trajectory(sol, frame, m_star, m_p, R_p, ...)` — unified trajectory analysis
- `extract_encounter_states(sol, m_p, R_p, ...)` → `EncounterGeometry` dataclass
- Data classes: `EncounterGeometry` (encounter metrics with reason codes)

**Monte Carlo** (`slingshot.analysis.monte_carlo`):
- `run_monte_carlo(N, t_span, m_star, m_p, frame, ...)` — full MC sweep with parallelization
- `select_top_indices(mc, metric, sign, top_frac, min_top)` — flexible candidate ranking

**Baselines** (`slingshot.analysis.baselines`):
- `compare_3body_with_baselines(sol, enc, m_star, m_p, R_p, ...)` — 2-body vs 3-body comparison

**Narrowed Baselines** (`slingshot.analysis.narrowed_baselines`):
- `compute_narrowed_baselines(analyses_top, cfg, padding_factor, ...)` — velocity-matched envelope sweeps
- Data classes: `BaselineResult`, `EnvelopeSummary`

**Comparison** (`slingshot.analysis.comparison`):
- `compare_2body_3body(energy_2body_star, energy_2body_planet, energy_3body, ...)` → dict
- `format_energy(value)` → readable string
- `print_comparison(comp)` → formatted output

### Output Layer (`slingshot.output`)

**Plotting 3-Body** (`slingshot.output.plotting`):
- `plot_mc_summary(mc)`, `plot_best_candidate_with_bodies(sol, ana, ...)`
- `plot_velocity_phase_space(sol)`, `plot_star_proximity_distribution(mc, ...)`
- `plot_planet_frame_diagnostics(analyses, ...)`, `plot_multi_candidate_overlay(sols, ...)`
- `plot_rejection_breakdown(mc)`, `plot_parameter_correlations(mc)`, `plot_energy_cdf(mc)`

**Plotting 2-Body** (`slingshot.output.plotting_twobody`):
- `plot_poincare_heatmaps(M_body_kg, v_inf_kms, ...)` — multi-panel Poincaré maps
- `plot_scattering_maps(M_body_kg, v_approach_kms, ...)` — scattering angle maps
- `plot_encounter_2d_cartesian(M_body_kg, ...)` — Cartesian encounter grids
- `plot_encounter_2d_trajectories(M_body_kg, ...)` — multi-scenario comparison
- `plot_trajectory_tracks(narrowed, sols_best, ...)` — 2B vs 3B overlay trajectories

**Animation** (`slingshot.output.animation`):
- `animate_trajectory(sol, ...)`, `animate_phase_space(sol, ...)`
- `generate_all_animations(sol, ...)` — orchestrate all video rendering

**Report** (`slingshot.output.report`):
- `generate_run_report(output_dir, cfg, mc, analyses, best, ...)` → write REPORT.md

**Compare Runs** (`slingshot.output.compare_runs`):
- `compare_runs(run_dirs)` — load and cross-compare multiple results
- `print_comparison(run_dirs)` — formatted comparison table

### Pipeline & Configuration

**Pipeline** (`slingshot.pipeline`):
- `run_pipeline(config_path, output_dir, phases, skip_plots, verbose)` — full 8-phase orchestrator
- Phase functions: `phase_monte_carlo`, `phase_select`, `phase_rerun`, `phase_best_selection`, `phase_baselines`, `phase_plots`, `phase_animations`, `phase_save`

**Configuration** (`slingshot.config`):
- `load_config(path)` → `FullConfig` (Pydantic v2 validated)
- `save_config(cfg, path)` — export to YAML
- Models: `SystemConfig`, `SamplingConfig`, `NumericalConfig`, `SelectionObjectiveConfig`, `PipelineConfig`, `VisualizationConfig`, `TwoBodyConfig`, `FullConfig`

**CLI** (`slingshot.cli`):
- `main()` — entry point for `slingshot` command
- Supports: `run`, `compare` subcommands with argparse

---

## Performance

**Typical runtime** (Kepler-432, N = 3000):

| Mode | Duration | Notes |
|------|----------|-------|
| 2-body scan (150 × 200) | 30–60 s | Single-threaded |
| 3-body MC (serial) | 2–3 hours | Single-threaded |
| 3-body MC (4 cores) | 40–50 min | `n_parallel: 4` |

---

## Troubleshooting

**ImportError: No module named 'slingshot'**

```bash
pip install -r requirements.txt
# or ensure you run from the workspace root
```

**ValidationError loading config** — Check YAML syntax and field names; see `slingshot/config.py` for the Pydantic schema.

**Old SI units (m-kg-s)** — All code is now km-kg-s. If values are 10⁶× too large, you may be using an outdated config. Energy: 1 km²/s² = 1 MJ/kg. Old J/kg results should be divided by 10⁶.

**Parallelisation not working** — Set `pipeline.n_parallel` to an integer > 1.

**Animation generation fails** — Ensure `ffmpeg` is installed (`ffmpeg -version`). Try GIF format as fallback.

---

## Requirements

```
numpy
scipy
matplotlib
pyyaml
pydantic
```

Optional: `ffmpeg-python` (video), `pandas` (tables), `pytest` (testing)

---

## v1 → v3.0 Improvements

| Feature | v1 | v2.4 | v3.0 |
|---------|-----|------|------|
| **Structure** | monolithic notebook | 17 flat modules + CLI | 3 layers (core/analysis/output) + packaging |
| **Packaging** | none | importable only | pip-installable + CLI entry point |
| **Unit system** | mixed | km-kg-s everywhere | km-kg-s everywhere |
| **Constants** | hardcoded (9+ places) | `constants.py` | `constants.py` + re-exports |
| **Architecture** | flat | modules | layered (core / analysis / output) |
| **Config** | hardcoded | YAML + Pydantic | YAML + Pydantic v2 |
| **Workflow** | manual cells | `python run.py` | `slingshot` CLI command |
| **2-body baselines** | star only | star+planet | narrowed envelopes (velocity-matched) |
| **Star filtering** | none | configurable R★ | configurable + reproducible |
| **Planet-frame** | none | EncounterGeometry | EncounterGeometry + v3.0 fields |
| **Diagnostic plots** | 3 | 14 (9+5) | 14 + trajectory tracks overlay |
| **Output org** | scattered | `results/` per-run | `results/` + proper gitignore |
| **Reporting** | manual | auto REPORT.md | auto REPORT.md + markdown lists |
| **Run comparison** | none | CLI subcommand | `slingshot compare` with tables |
| **Duplication** | 400+ lines | eliminated | eliminated + DRY imports |
| **Parallelisation** | none | ProcessPoolExecutor | ProcessPoolExecutor |
| **Animation** | none | trajectory+phase-space | trajectory + phase-space video |
| **Cross-comparison** | placeholder | `comparison.py` | `comparison.py` formatted output |
| **Error diagnostics** | minimal | reason codes | EncounterGeometry + reason codes |
| **Tests** | none | none | 29 tests (core, analysis, config) |
| **Import paths** | flat | `slingshot.*` | `slingshot.core.*`, `.analysis.*`, `.output.*` + top-level re-exports |

---

## License

[Specify your license here]

---

**Version**: 3.0.0  
**Last Updated**: February 2026  
**Status**: Production-ready with comprehensive test suite (29 tests)
