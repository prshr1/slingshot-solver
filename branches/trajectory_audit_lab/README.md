# Trajectory Audit Lab

This workspace is intentionally separated from the main production pipeline.
Use it for experimental trajectory rendering, audit diagnostics, and test harnesses.

## Structure

- `notebooks/` : exploratory and standalone notebooks
- `scripts/` : reusable experimental scripts
- `audits/` : audit outputs, reports, and generated diagnostics
- `tests/` : branch-local tests and validation checks

## Current Notebook

- `notebooks/TrajectoryGradientsLatest.ipynb`

## Scripts

- `scripts/trajectory_gradient_renderer.py` : render static/video trajectory gradients from the latest (or selected) run record
- `scripts/relocate_experiment_outputs.py` : move audit/smoke/temp result directories into this lab

## Quick Commands

- Static render from latest run:
  `python branches/trajectory_audit_lab/scripts/trajectory_gradient_renderer.py --mode static`
- Both static + video from latest run:
  `python branches/trajectory_audit_lab/scripts/trajectory_gradient_renderer.py --mode both`
- Relocate existing audit/smoke directories:
  `python branches/trajectory_audit_lab/scripts/relocate_experiment_outputs.py --execute`

## Workflow

1. Build/iterate in this lab first.
2. Validate via `tests/` and audit outputs in `audits/`.
3. Promote stable changes into production modules only after verification.
