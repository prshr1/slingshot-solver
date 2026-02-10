# Changelog

All notable changes to the Slingshot Solver project.

---

## v2.4.0 — February 2026

### Unified Pipeline & Report Auto-Generation

One-command workflow: `python run.py config.yaml` runs the full 8-phase pipeline and produces a complete results directory with auto-generated `REPORT.md`. All 6 standalone 2-body visualisation scripts absorbed into the package as config-driven functions.

**New modules**:
- `slingshot/pipeline.py` — 8-phase orchestrator (`run_pipeline()`) with independently callable phase functions (`phase_monte_carlo`, `phase_select`, `phase_rerun`, `phase_best_selection`, `phase_baselines`, `phase_plots`, `phase_animations`, `phase_save`).
- `slingshot/report.py` — `generate_run_report()` produces a markdown analysis report with system params, MC statistics, rejection breakdown, best candidates, 2-body vs 3-body comparison, and saved plots listing.
- `slingshot/compare_runs.py` — `compare_runs()` and `print_comparison()` for cross-run comparison tables. Loads `config.yaml` + `summary.csv` from each results directory.
- `slingshot/plotting_twobody.py` — absorbed 6 standalone scripts into 5 config-driven functions:
  - `plot_poincare_heatmaps()` — multi-panel Poincaré deflection/ΔV heatmaps
  - `plot_scattering_maps()` — scattering angle maps at multiple approach angles
  - `plot_encounter_2d_cartesian()` — Cartesian encounter grid visualisation
  - `plot_encounter_2d_trajectories()` — multi-scenario trajectory comparison
  - `plot_oberth_comparison()` — no-burn vs Oberth manoeuvre with gain analysis
- `run.py` — CLI entry point with argparse. Default: `python run.py <config>`. Subcommand: `python run.py compare <dir1> <dir2> ...`. Options: `--output-dir`, `--skip-plots`, `--skip-animations`, `--phases`, `--quiet`.

**Expanded modules**:
- `slingshot/plotting.py` — 6 new diagnostic plot functions:
  - `plot_star_proximity_distribution()` — histogram of r_star_min / R★
  - `plot_planet_frame_diagnostics()` — 4-panel planet-frame bar charts (Δv, deflection, energy, star proximity)
  - `plot_multi_candidate_overlay()` — top-N trajectories on single figure
  - `plot_rejection_breakdown()` — horizontal bar chart of rejection reasons
  - `plot_parameter_correlations()` — 4-panel scatter matrix (ΔV vs deflection, r_min, star proximity)
  - `plot_energy_cdf()` — CDF of ½|ΔV_vec|² with percentile markers
- `slingshot/config.py` — 8 new fields in `VisualizationConfig`: `generate_2body_heatmaps`, `generate_scattering_maps`, `generate_poincare_maps`, `generate_oberth_maps`, `generate_trajectory_heatmap`, `heatmap_grid_resolution`, `heatmap_approach_angles_deg`, `top_n_overlay`.
- `slingshot/__init__.py` — version bumped to 2.4.0, all new exports added.

**Archived (moved to `Archive/`)**:
- 7 standalone scripts: `encounter_2d_cartesian.py`, `encounter_2d_trajectories.py`, `oberth_poincare.py`, `poincare_heatmap.py`, `scattering_maps.py`, `trajectory_heatmap_2d.py`, `trajectory_tracks.py`
- `KippingCase/` directory

**Documentation**:
- README.md updated to v2.4 with new Quick Start (`run.py` CLI), architecture diagram, pipeline data flow, expanded API reference, v1→v2.4 comparison table.
- CHANGELOG.md updated with v2.4.0 entry.

---

## v2.3.0 — February 9, 2026

### Star-Proximity Filtering & Interstellar Config

Added physical validity enforcement: trajectories penetrating the stellar surface are now rejected during Monte Carlo. New interstellar-velocity configuration for Kepler-432.

**New features**:
- `star_min_clearance_Rstar` parameter in `NumericalConfig` — rejects trajectories closer than N × R★ to the star.
- `R_star_Rsun` field in `SystemConfig` — stellar radius for clearance checks.
- Planet-frame encounter diagnostics in `analysis.py` — `EncounterGeometry` now includes planet-relative state vectors.
- Interstellar config (`configs/config_interstellar_k432.yaml`) — v∞ = 5–200 km/s, b = 0.0001–0.005 AU, 24,000 particles.
- All notebook plot cells now auto-save figures to the per-run results directory.

**Workspace reorganisation**:
- Configs moved to `configs/` directory.
- All run output goes to `results/results_{system}_{timestamp}/` with config, data, and plots.
- Standalone script PNGs save to `results/figures/`.
- Animation frames save to `results/frames/`.
- Deprecated files archived to `Archive/`.
- Added `.gitignore` (ignores `results/`, `Archive/`, `__pycache__/`, `*.png`).

**Modified modules**:
- `monte_carlo.py` — enforces `star_min_clearance_Rstar` filter; new rejection reason `"star_penetration"`.
- `config.py` — added `star_min_clearance_Rstar`, `R_star_Rsun` fields.
- `analysis.py` — planet-frame diagnostics added to `EncounterGeometry`.
- `animation.py` — default `output_dir` changed from `"./frames"` to `"./results/frames"`.
- All 7 standalone scripts — output PNGs now go to `results/figures/`.
- `trajectory_tracks.py` — default config path updated to `configs/`.

**Notebook updates**:
- `ThreeBodySolver_v2.ipynb` — config path updated to `configs/`, output_dir created early (after MC), 5 plot cells now auto-save to run directory, baselines plot saves via `plot_save_dir`.

**Documentation**:
- README.md updated to v2.3 with new directory structure, updated paths, and feature table.
- `REPORT.md` — detailed run analysis from Feb 9 (24,000 particles, 1,010 successful, best ΔV +24.22 km/s).

---

## v2.1.0 — February 7, 2026

### Unit System Unification

Migrated the entire codebase from mixed m-kg-s / km-kg-s to a single **km-kg-s** unit system.

**New modules**:
- `slingshot/constants.py` — single source of truth for G_KM, M_SUN, M_JUP, R_JUP, R_SUN, AU_KM, plus helper functions (`mu_star`, `mu_planet`, `au_to_km`).
- `slingshot/comparison.py` — `compare_2body_3body()`, `format_energy()`, `print_comparison()` for cross-solver analysis with consistent units (km²/s² ≡ MJ/kg).

**Rewritten modules**:
- `slingshot/twobody.py` — full rewrite in km-kg-s. `EncounterGeometry` renamed to `TwoBodyGeometry` to avoid collision with `analysis.EncounterGeometry`. Added `create_planet_encounter_from_config()` factory for planet-scattering baselines.
- `trajectory_tracks.py` — rewritten to use `FullConfig` (Pydantic), `run_2body_analysis()` entry point, supports `scattering_body: "both"` for dual baselines.

**Modified modules** (constants replaced with imports from `constants.py`):
- `dynamics.py`, `analysis.py`, `baselines.py`, `sampling.py`, `monte_carlo.py`, `plotting.py`

**Config changes**:
- `config_kepler432_case.yaml` — rewritten to flat schema matching `FullConfig`. All distances now in km. Physical constants removed from YAML (live in code). Added `two_body` section with `scattering_body: "both"`, `TwoBodyConfig` model added to `config.py`.
- `FullConfig` updated with `extra="ignore"` so the `two_body` block does not cause validation errors.
- Removed unused `PhysicalConstants` dataclass from `config.py`.

**Notebook updates**:
- `Kepler432_Integration.ipynb` — complete rewrite for v2.1 unified API. Now runs both 2-body baselines (star + planet) and 3-body MC in a single notebook, with energy CDF overlay and cross-solver comparison.
- `ThreeBodySolver_v2.ipynb` — removed hardcoded M_SUN/M_JUP/R_JUP/AU_KM; imports from `slingshot.constants`. Removed debug reload cells.

**Package exports** (`__init__.py`):
- Version bumped to 2.1.0
- Added exports: `G_KM`, `M_SUN`, `M_JUP`, `R_JUP`, `R_SUN`, `AU_KM`, `mu_star`, `mu_planet`, `au_to_km`, `TwoBodyEncounter`, `TwoBodyGeometry`, `TrajectoryResult`, `create_encounter_from_config`, `create_planet_encounter_from_config`, `compare_2body_3body`, `format_energy`, `print_comparison`, `TwoBodyConfig`, `VisualizationConfig`.

**Documentation**:
- Consolidated ARCHITECTURE.md, QUICKSTART.md, IMPLEMENTATION_SUMMARY.md, IMPLEMENTATION_README.md, IMPLEMENTATION_CHECKLIST.md, FIX_SUMMARY.md into a single README.md.
- CHANGELOG.md streamlined to a proper changelog format.

### Breaking Changes

- `trajectory_tracks.py` — removed `load_config_yaml()`, `extract_2body_parameters()`. Use `run_2body_analysis()` or `create_trajectory_tracks_from_config()` instead.
- `slingshot/twobody.py` — `EncounterGeometry` class renamed to `TwoBodyGeometry`. Energy/distance outputs now in km-kg-s (not m-kg-s).
- Config YAML schema changed from deeply nested to flat keys.

### Bug Fixes

- Fixed `VisualizationConfig` listed in `__all__` but never imported in `__init__.py`.
- Fixed silent config defaults — old nested YAML keys were ignored by Pydantic, causing all parameters to use defaults.
- Eliminated 9+ duplicate constant definitions across the package.

---

## v2.0.0 — February 3, 2026

### New Features

- **Modular package structure**: Refactored 1,200-line monolithic notebook into 9 focused Python modules.
- **Configuration system**: YAML/JSON with Pydantic validation (no hardcoded parameters). `load_config()`, `save_config()`, predefined system configs.
- **Unified Monte Carlo**: Merged `run_batch_mc_3body()` and `run_batch_mc_3body_barycentric()` into single `run_monte_carlo()` with `frame` parameter.
- **Unified analysis**: Merged two analysis functions into `analyze_trajectory(frame="planet"|"barycentric")`.
- **Robust encounter extraction**: `EncounterGeometry` dataclass with `.ok` flag and `.reason` diagnostic string.
- **Parallelisation**: `ProcessPoolExecutor` support via `n_parallel` parameter (3–4× speedup).
- **Animation/video rendering**: `animate_trajectory()` and `animate_phase_space()` (MP4/GIF).
- **Flexible candidate selection**: `select_top_indices()` with configurable metrics and direction.

### Module Breakdown

| Module | LOC | Responsibility |
|--------|-----|----------------|
| `config.py` | 350 | Configuration management (Pydantic) |
| `dynamics.py` | 180 | 3-body ODE and integration |
| `analysis.py` | 350 | Trajectory analysis, encounter extraction |
| `sampling.py` | 200 | Initial condition generation |
| `monte_carlo.py` | 280 | MC orchestration and parallelisation |
| `baselines.py` | 320 | 2-body hyperbola and monopole baselines |
| `plotting.py` | 220 | Static visualisation |
| `animation.py` | 300 | Video rendering |

### Breaking Changes

- API completely refactored; no backward compatibility with v1.
- Original notebook (`ThreeBodySolver3.ipynb`) preserved as reference only.

---

## v1.0.0 — Original Implementation

Monolithic `ThreeBodySolver3.ipynb` notebook. Functional but with hardcoded parameters, duplicated code, and no modular structure. See `ThreeBodySolver3.ipynb` for reference.

---

## Future Roadmap

- [ ] 3D orbital dynamics (z-coordinate)
- [ ] Eccentric orbits for star-planet binary
- [ ] GPU ODE integration (JAX/CuPy)
- [ ] Multi-trajectory comparison animations (Type C)
- [ ] ML-based outcome prediction
- [ ] Interactive parameter tuning (ipywidgets)
- [ ] Statistical uncertainty quantification
