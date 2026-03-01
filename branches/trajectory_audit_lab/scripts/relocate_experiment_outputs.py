#!/usr/bin/env python
"""Relocate existing experimental artifacts into trajectory_audit_lab.

Moves audit-like directories out of results/ into branch-local audit/test locations.
Default mode is dry-run. Use --execute to perform moves.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_BRANCH_ROOT = Path("branches/trajectory_audit_lab")


@dataclass
class MovePlan:
    source: Path
    destination: Path
    category: str


def classify_results_dir(name: str) -> Optional[str]:
    if name.startswith("audit_"):
        return "audit_history"
    if name.startswith("_smoke") or name.startswith("_diag") or name.startswith("_tmp"):
        return "test_artifacts"
    if name in {"figures", "frames"}:
        return "audit_history"
    return None


def build_move_plans(results_root: Path, branch_root: Path) -> list[MovePlan]:
    plans: list[MovePlan] = []
    if not results_root.exists():
        return plans

    for src in sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        category = classify_results_dir(src.name)
        if category is None:
            continue

        if category == "audit_history":
            dst = branch_root / "audits" / "history" / src.name
        else:
            dst = branch_root / "tests" / "artifacts" / src.name

        plans.append(MovePlan(source=src, destination=dst, category=category))

    return plans


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    idx = 1
    while True:
        candidate = path.with_name(f"{path.name}_moved_{idx:02d}")
        if not candidate.exists():
            return candidate
        idx += 1


def execute_plans(plans: Iterable[MovePlan], execute: bool) -> list[str]:
    logs: list[str] = []
    for plan in plans:
        dst = unique_destination(plan.destination)
        logs.append(f"[{plan.category}] {plan.source} -> {dst}")
        if execute:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(plan.source), str(dst))
    return logs


def write_manifest(branch_root: Path, logs: list[str], dry_run: bool) -> Path:
    reports_dir = branch_root / "audits" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mode = "dry_run" if dry_run else "executed"
    manifest = reports_dir / f"relocation_manifest_{mode}.md"

    lines = [
        "# Relocation Manifest",
        "",
        f"Mode: {'DRY RUN' if dry_run else 'EXECUTED'}",
        "",
        "## Moves",
        "",
    ]
    if logs:
        lines.extend([f"- {line}" for line in logs])
    else:
        lines.append("- No matching directories found.")

    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Relocate experimental output directories into branch workspace.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--branch-root", type=Path, default=DEFAULT_BRANCH_ROOT)
    p.add_argument("--execute", action="store_true", help="Perform moves (default is dry-run)")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    plans = build_move_plans(args.results_root, args.branch_root)
    logs = execute_plans(plans, execute=bool(args.execute))
    manifest = write_manifest(args.branch_root, logs, dry_run=not args.execute)

    print(f"Results root: {args.results_root}")
    print(f"Branch root: {args.branch_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Planned/processed moves: {len(logs)}")
    for line in logs:
        print(f" - {line}")
    print(f"Manifest: {manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
