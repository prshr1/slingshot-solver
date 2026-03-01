# Slingshot Solver v2.4

**Modular Python package for studying gravitational slingshot (gravity-assist) dynamics in restricted 3-body systems, with a focus on interstellar-velocity encounters.**

Star + Hot Jupiter + Satellite orbital mechanics with Monte Carlo simulations, 2-body baselines, star-proximity filtering, planet-frame diagnostics, auto-generated reports, and video animation.

**Canonical unit system**: km-kg-s throughout. Energies in km²/s² (≡ MJ/kg).

---

## Quick Start

### Install

```bash
cd slingshot-solver
pip install -r requirements.txt
```

### One-Command Pipeline (recommended)

```bash
python run.py configs/config_kepler432_case.yaml
```

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

```
slingshot-solver/
├── configs/                        # YAML configuration files
│   ├── config_default.yaml
│   ├── config_interstellar_k432.yaml
│   ├── config_kepler432_case.yaml
│   ├── config_pub_methodology_single.yaml
│   ├── config_pub_methodology_pareto.yaml
│   └── config_pub_methodology_weighted.yaml
├── results/                        # All outputs (gitignored)
│   └── results_Kepler-432_*/       # Per-run dirs (config, plots, data, REPORT.md)
├── slingshot/                      # Core Python package
│   ├── __init__.py
│   ├── constants.py                # G_KM, M_SUN, M_JUP, R_JUP, R_SUN, AU_KM
│   ├── config.py                   # Pydantic config models + YAML loader
│   ├── dynamics.py                 # 3-body ODE + RK integration
│   ├── analysis.py                 # Trajectory analysis + EncounterGeometry
│   ├── sampling.py                 # Initial condition generation
│   ├── monte_carlo.py              # Monte Carlo sweep + candidate selection
│   ├── baselines.py                # 2-body hyperbola + monopole baselines
│   ├── narrowed_baselines.py       # Post-hoc narrowed 2-body comparisons
│   ├── twobody.py                  # TwoBodyEncounter grid-scan class
│   ├── comparison.py               # 2-body vs 3-body cross-comparison
│   ├── plotting.py                 # 3-body diagnostic plots (9 functions)
│   ├── plotting_twobody.py         # 2-body heatmaps & encounter maps (5 functions)
│   ├── animation.py                # Video rendering (trajectory + phase-space)
│   ├── pipeline.py                 # 8-phase orchestrator
│   ├── report.py                   # Auto-generated REPORT.md
│   └── compare_runs.py             # Cross-run comparison tables
├── run.py                          # CLI entry point
├── ThreeBodySolver_v2.ipynb        # Interactive notebook
├── TwoBodyScatter.py               # Closed-form hyperbolic solver
├── Archive/                        # Deprecated standalone scripts (gitignored)
├── REPORT.md                       # Latest run analysis
├── CHANGELOG.md
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
python run.py configs/config_kepler432_case.yaml
```

Runs all 8 phases: MC → Select → Rerun → Best → Baselines → Plots → Animations → Save. Produces a timestamped results directory with `REPORT.md`, all diagnostic plots, and pickled data.

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

Config-driven 2-body encounter visualisations (Poincaré maps, scattering maps, Cartesian encounters, Oberth comparison):

```python
from slingshot.plotting_twobody import plot_poincare_heatmaps, plot_scattering_maps

figs = plot_poincare_heatmaps(
    M_body_kg=5.2 * 1.898e27,
    v_inf_kms=30.0,
    vstar0_kms=0.0,
    body_label="Planet",
    save_dir="results/figures",
)
```

These are also auto-generated by the pipeline when `visualization.generate_2body_heatmaps: true` in the config.

### 4. Cross-Run Comparison

```python
from slingshot.compare_runs import compare_runs, print_comparison

print_comparison(["results/run_a", "results/run_b"])
```

Or from the CLI:

```bash
python run.py compare results/run_a results/run_b
```

### 5. Animations

```python
from slingshot.animation import generate_all_animations

animations = generate_all_animations(
    best_sol,
    output_dir="./results/frames",
    video_fps=30,
    video_format="mp4",
)
```

---

## API Reference

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

### Dynamics (`slingshot.dynamics`)

- `simulate_3body(Y0, t_span, m_star, m_p, n_eval, rtol, atol)` — RK45 integration
- `init_hot_jupiter_barycentric(a_km, m_star, m_p, phase, prograde)` — initial conditions

### Analysis (`slingshot.analysis`)

- `analyze_trajectory(sol, frame, m_star, m_p, R_p, ...)` — unified trajectory analysis
- `extract_encounter_states(sol, m_p, R_p, ...)` → `EncounterGeometry`

### Monte Carlo (`slingshot.monte_carlo`)

- `run_monte_carlo(N, t_span, m_star, m_p, frame, ...)` — full MC sweep
- `select_top_indices(mc, metric, sign, top_frac)` — flexible ranking

### Two-Body (`slingshot.twobody`)

- `TwoBodyEncounter(M_body_kg, G, label)` — encounter manager
- `TwoBodyEncounter.scan_parameter_space(...)` — grid scan over b × angle
- `TwoBodyEncounter.get_energy_statistics(energies)` — distribution stats
- `create_encounter_from_config(cfg)` — star scattering factory
- `create_planet_encounter_from_config(cfg)` — planet scattering factory
- Data classes: `TwoBodyGeometry`, `TrajectoryResult`

### Comparison (`slingshot.comparison`)

- `compare_2body_3body(...)` → dict with ΔE, % improvement
- `format_energy(value)` → readable string
- `print_comparison(comp)` → formatted output

### Visualisation (`slingshot.plotting`, `slingshot.plotting_twobody`, `slingshot.animation`)

**3-body diagnostics** (`plotting`):
- `plot_mc_summary(mc)` — scatter + histogram overview
- `plot_best_candidate_with_bodies(sol, ana, ...)` — trajectory with bodies
- `plot_velocity_phase_space(sol)` — velocity evolution
- `plot_star_proximity_distribution(mc, R_star_km, ...)` — r_min/R★ histogram
- `plot_planet_frame_diagnostics(analyses, ...)` — 4-panel planet-frame bar charts
- `plot_multi_candidate_overlay(sols, analyses, ...)` — top-N trajectories overlay
- `plot_rejection_breakdown(mc, ...)` — horizontal bar chart of rejection reasons
- `plot_parameter_correlations(mc, ...)` — 4-panel scatter matrix
- `plot_energy_cdf(mc, ...)` — CDF of ½|ΔV_vec|² with percentile markers

**2-body encounter maps** (`plotting_twobody`):
- `plot_poincare_heatmaps(M_body_kg, v_inf_kms, ...)` — multi-panel Poincaré heatmaps
- `plot_scattering_maps(M_body_kg, v_approach_kms, ...)` — scattering angle maps
- `plot_encounter_2d_cartesian(M_body_kg, ...)` — Cartesian encounter grids
- `plot_encounter_2d_trajectories(M_body_kg, ...)` — multi-scenario trajectory comparison
- `plot_oberth_comparison(M_body_kg, ...)` — no-burn vs Oberth manoeuvre comparison

**Animation** (`animation`):
- `animate_trajectory(sol, ...)`, `animate_phase_space(sol, ...)`
- `generate_all_animations(sol, ...)` — orchestrate all video types

### Pipeline (`slingshot.pipeline`)

- `run_pipeline(config_path, output_dir, phases, skip_plots, skip_animations, verbose)` — full 8-phase orchestrator
- `phase_monte_carlo(cfg, verbose)` — Monte Carlo sweep
- `phase_select(cfg, mc, verbose)` — candidate ranking
- `phase_rerun(cfg, mc, top_idx, verbose)` — re-integrate top candidates
- `phase_best_selection(analyses, top_idx, sols, verbose)` — find best by scalar/vector ΔV
- `phase_baselines(cfg, analyses, best_ana, verbose)` — 2-body comparison baselines
- `phase_plots(cfg, mc, top_idx, rerun, best, output_dir, verbose)` — all diagnostic plots
- `phase_animations(cfg, best_sol, output_dir, verbose)` — video rendering
- `phase_save(cfg, mc, top_idx, rerun, best, baselines, output_dir, verbose)` — persist results

### Report & Comparison (`slingshot.report`, `slingshot.compare_runs`)

- `generate_run_report(output_dir, cfg, mc, analyses, best, comparison, narrowed, plots)` — auto-generate REPORT.md
- `compare_runs(run_dirs)` — load and cross-compare multiple result directories
- `print_comparison(run_dirs)` — formatted comparison table

### Configuration (`slingshot.config`)

- `load_config(path)` → `FullConfig` (Pydantic-validated)
- `save_config(cfg, path)` — export to YAML
- Models: `SystemConfig`, `SamplingConfig`, `NumericalConfig`, `SelectionObjectiveConfig`, `PipelineConfig`, `VisualizationConfig`, `TwoBodyConfig`, `FullConfig`

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

## v1 → v2.4 Improvements

| Feature | v1 | v2.4 |
|---------|-----|------|
| Structure | monolithic notebook | 17 modules + CLI |
| Units | mixed (m-kg-s / km-kg-s) | km-kg-s everywhere |
| Constants | hardcoded in 9+ places | single `constants.py` |
| Config | hardcoded | YAML + Pydantic validation |
| Workflow | manual cell-by-cell | `python run.py config.yaml` (one command) |
| 2-body baselines | star only | star + planet + both |
| 2-body visualisation | 6 standalone scripts | absorbed into `plotting_twobody.py` |
| Star filtering | none | configurable R★ clearance |
| Planet-frame diagnostics | none | EncounterGeometry with planet reference |
| Diagnostic plots | 3 | 14 (9 three-body + 5 two-body) |
| Output organisation | root directory | `results/` with per-run dirs |
| Reporting | manual | auto-generated REPORT.md |
| Run comparison | none | `compare_runs.py` + CLI subcommand |
| Duplication | ~400 lines | eliminated |
| Parallelisation | none | ProcessPoolExecutor |
| Animation | none | trajectory + phase-space video |
| Cross-solver comparison | placeholder | `comparison.py` with formatted output |
| Error diagnostics | minimal | EncounterGeometry with reason codes |

---

## License

[Specify your license here]

---

**Version**: 2.4.0  
**Last Updated**: February 2026
