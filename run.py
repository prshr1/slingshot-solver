#!/usr/bin/env python
"""
Slingshot Solver — backward-compatible CLI entry point.

This thin wrapper delegates to slingshot.cli.main().
Prefer using the installed entry point: ``slingshot <config.yaml>``

Usage:
    python run.py <config.yaml>                           # Full pipeline
    python run.py <config.yaml> --output-dir DIR          # Custom output
    python run.py <config.yaml> --skip-plots              # No figures
    python run.py <config.yaml> --phases mc,rerun         # Subset of phases
    python run.py compare results/run_a results/run_b     # Cross-run comparison
    python run.py --compare results/run_a results/run_b   # Cross-run comparison
"""

from slingshot.cli import main

if __name__ == "__main__":
    main()
