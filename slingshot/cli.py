#!/usr/bin/env python
"""
Slingshot Solver CLI entry point.

Usage:
    slingshot <config.yaml>                           # Full pipeline
    slingshot <config.yaml> --output-dir DIR          # Custom output
    slingshot <config.yaml> --skip-plots              # No figures
    slingshot <config.yaml> --phases mc,rerun         # Subset of phases
    slingshot compare results/run_a results/run_b     # Cross-run comparison
"""

import argparse
import sys
from pathlib import Path

from .console import configure_console_streams, safe_print


def _run_compare(run_dirs):
    from .output.compare_runs import print_comparison

    if not run_dirs:
        safe_print("Error: compare requires at least one run directory")
        safe_print("Usage: slingshot compare <run_dir> [<run_dir> ...]")
        sys.exit(2)

    print_comparison(run_dirs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slingshot gravitational-assist pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("config", nargs="?", help="Path to YAML config file")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_DIR",
        help="Compare one or more result directories",
    )
    parser.add_argument("--output-dir", "-o", default=None, help="Override output directory")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plot generation")
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Skip animation/video generation",
    )
    parser.add_argument(
        "--phases",
        default=None,
        help="Comma-separated phases: mc,select,rerun,best,baselines,plots,animations,save",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    return parser


def main(argv=None):
    """Main CLI entry point for slingshot-solver."""
    configure_console_streams()

    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward-compatible subcommand form:
    # slingshot compare <run_dir> [<run_dir> ...]
    if argv and argv[0] == "compare":
        _run_compare(argv[1:])
        return

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.compare is not None:
        _run_compare(args.compare)
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    config_path = args.config
    if not Path(config_path).exists():
        safe_print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    phases = args.phases.split(",") if args.phases else None

    from .pipeline import run_pipeline

    results = run_pipeline(
        config_path=config_path,
        output_dir=args.output_dir,
        phases=phases,
        skip_plots=args.skip_plots,
        skip_animations=args.skip_animations,
        verbose=not args.quiet,
    )

    if not args.quiet and "output_dir" in results:
        safe_print(f"\n  Results: {results['output_dir']}")


if __name__ == "__main__":
    main()
