#!/usr/bin/env python
"""Render trajectory gradients from the latest or selected record.

Outputs static and/or animated artifacts for:
- star
- planet
- star+planet combined views

This script is branch-local and intended for audit/experimentation workflows.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Ensure repository root is importable when running this script via path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slingshot.config import load_config
from slingshot.narrowed_baselines import compute_narrowed_baselines
from slingshot.plotting_twobody import plot_trajectory_tracks


DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_OUTPUT_ROOT = Path("branches/trajectory_audit_lab/audits/generated")


def find_latest_run_dir(results_root: Path) -> Path:
    """Find the latest run dir containing both results.pkl and config.yaml."""
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    candidates = [
        p
        for p in results_root.iterdir()
        if p.is_dir() and (p / "results.pkl").exists() and (p / "config.yaml").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with results.pkl + config.yaml under: {results_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_output_dir(output_root: Path, run_dir: Path, output_tag: Optional[str]) -> Path:
    """Resolve branch-local output directory."""
    tag = output_tag.strip() if output_tag else run_dir.name
    out = output_root / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_run_context(run_dir: Path) -> dict[str, Any]:
    """Load config + record and compute narrowed baselines."""
    cfg = load_config(str(run_dir / "config.yaml"))

    with open(run_dir / "results.pkl", "rb") as f:
        record = pickle.load(f)

    rerun = record.get("rerun", {}) if isinstance(record, dict) else {}
    sols_best = record.get("sols_best", []) or rerun.get("sols_best", []) or []
    analyses_best = record.get("analyses_best", []) or rerun.get("analyses_best", []) or []
    valid_analyses = [a for a in analyses_best if a is not None]

    if not valid_analyses:
        raise RuntimeError("No valid analyses found in results.pkl; cannot build narrowed baselines.")

    tb_cfg = getattr(cfg, "two_body", None)
    narrowed = compute_narrowed_baselines(
        analyses_top=valid_analyses,
        cfg=cfg,
        padding_factor=tb_cfg.padding_factor if tb_cfg else 1.5,
        num_v=tb_cfg.num_v if tb_cfg else 20,
        num_b=tb_cfg.num_b_narrow if tb_cfg else 100,
        num_angles=tb_cfg.num_angles_narrow if tb_cfg else 100,
        verbose=False,
    )

    return {
        "cfg": cfg,
        "record": record,
        "sols_best": sols_best,
        "analyses_best": analyses_best,
        "narrowed": narrowed,
    }


def save_static_combined(output_dir: Path, figsize: Tuple[float, float], dpi: int) -> Path:
    star_path = output_dir / "trajectory_tracks_star.png"
    planet_path = output_dir / "trajectory_tracks_planet.png"

    if not star_path.exists() or not planet_path.exists():
        missing = [str(p) for p in (star_path, planet_path) if not p.exists()]
        raise FileNotFoundError(f"Missing static input image(s): {missing}")

    img_star = plt.imread(star_path)
    img_planet = plt.imread(planet_path)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    axes[0].imshow(img_star)
    axes[0].set_title("Star Gradient Trajectory Tracks", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_planet)
    axes[1].set_title("Planet Gradient Trajectory Tracks", fontweight="bold")
    axes[1].axis("off")

    out_path = output_dir / "trajectory_tracks_star_planet_combined.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_time_npz(npz_path: Path) -> dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ time data: {npz_path}")
    d = np.load(npz_path)
    return {
        "energy_frames": d["energy_frames"],
        "count_frames": d["count_frames"],
        "x_edges_km": d["x_edges_km"],
        "y_edges_km": d["y_edges_km"],
        "energy_vmin": float(d["energy_vmin"]),
        "energy_vmax": float(d["energy_vmax"]),
        "body": str(d["body"]) if "body" in d else "unknown",
    }


def make_single_body_gif(
    npz_path: Path,
    out_path: Path,
    title: str,
    fps: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    data = load_time_npz(npz_path)
    ef = data["energy_frames"]
    x_edges = data["x_edges_km"] / 1e6
    y_edges = data["y_edges_km"] / 1e6
    n_frames = int(ef.shape[0])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(
        ef[0].T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=data["energy_vmin"],
        vmax=data["energy_vmax"],
        aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Scattering energy 0.5|dV|^2 (km^2/s^2)")

    frame_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=11,
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 3, "edgecolor": "none"},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X relative (10^6 km)")
    ax.set_ylabel("Y relative (10^6 km)")

    def _update(i: int):
        im.set_data(ef[i].T)
        frame_text.set_text(f"Frame {i + 1}/{n_frames}")
        return im, frame_text

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=1000.0 / max(fps, 1), blit=False)
    ani.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def make_combined_gif(
    star_npz: Path,
    planet_npz: Path,
    out_path: Path,
    fps: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    ds = load_time_npz(star_npz)
    dp = load_time_npz(planet_npz)

    ef_s = ds["energy_frames"]
    ef_p = dp["energy_frames"]
    n_frames = int(min(ef_s.shape[0], ef_p.shape[0]))

    extent_s = [
        ds["x_edges_km"][0] / 1e6,
        ds["x_edges_km"][-1] / 1e6,
        ds["y_edges_km"][0] / 1e6,
        ds["y_edges_km"][-1] / 1e6,
    ]
    extent_p = [
        dp["x_edges_km"][0] / 1e6,
        dp["x_edges_km"][-1] / 1e6,
        dp["y_edges_km"][0] / 1e6,
        dp["y_edges_km"][-1] / 1e6,
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    im_s = axes[0].imshow(
        ef_s[0].T,
        origin="lower",
        extent=extent_s,
        cmap="viridis",
        vmin=ds["energy_vmin"],
        vmax=ds["energy_vmax"],
        aspect="equal",
    )
    axes[0].set_title("Star Gradient Evolution", fontweight="bold")
    axes[0].set_xlabel("X relative (10^6 km)")
    axes[0].set_ylabel("Y relative (10^6 km)")
    cb_s = fig.colorbar(im_s, ax=axes[0], pad=0.01)
    cb_s.set_label("0.5|dV|^2")

    im_p = axes[1].imshow(
        ef_p[0].T,
        origin="lower",
        extent=extent_p,
        cmap="viridis",
        vmin=dp["energy_vmin"],
        vmax=dp["energy_vmax"],
        aspect="equal",
    )
    axes[1].set_title("Planet Gradient Evolution", fontweight="bold")
    axes[1].set_xlabel("X relative (10^6 km)")
    axes[1].set_ylabel("Y relative (10^6 km)")
    cb_p = fig.colorbar(im_p, ax=axes[1], pad=0.01)
    cb_p.set_label("0.5|dV|^2")

    frame_text = fig.text(0.5, 0.02, "", ha="center", va="center", fontsize=11)

    def _update(i: int):
        im_s.set_data(ef_s[i].T)
        im_p.set_data(ef_p[i].T)
        frame_text.set_text(f"Star + Planet Gradient Evolution | Frame {i + 1}/{n_frames}")
        return im_s, im_p, frame_text

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=1000.0 / max(fps, 1), blit=False)
    ani.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def render_static(
    output_dir: Path,
    ctx: dict[str, Any],
    gradient_mode: str,
    dpi: int,
    figsize: Tuple[float, float],
    num_b: int,
    num_angles: int,
    num_points: int,
    padding_frac: float,
    max_overlay_tracks: int,
    overlay_lines: bool,
    overlay_line_count: int,
    confidence_min_count: int,
    fixed_energy_range: Optional[Tuple[float, float]],
    hexbin_gridsize: int,
    kde_sigma_bins: float,
    time_frames: int,
) -> list[Path]:
    figs = plot_trajectory_tracks(
        narrowed=ctx["narrowed"],
        sols_best=ctx["sols_best"],
        analyses_best=ctx["analyses_best"],
        cfg=ctx["cfg"],
        num_b=num_b,
        num_angles=num_angles,
        num_points=num_points,
        padding_frac=padding_frac,
        max_overlay_tracks=max_overlay_tracks,
        overlay_lines=overlay_lines,
        overlay_line_count=overlay_line_count,
        gradient_mode=gradient_mode,
        confidence_min_count=confidence_min_count,
        fixed_energy_range=fixed_energy_range,
        hexbin_gridsize=hexbin_gridsize,
        kde_sigma_bins=kde_sigma_bins,
        time_frames=time_frames,
        export_phase_data=True,
        export_time_data=False,
        save_dir=str(output_dir),
        dpi=dpi,
    )
    for fig in figs:
        plt.close(fig)

    generated = [
        output_dir / "trajectory_tracks_star.png",
        output_dir / "trajectory_tracks_planet.png",
        save_static_combined(output_dir, figsize=figsize, dpi=dpi),
    ]
    return generated


def render_video(
    output_dir: Path,
    ctx: dict[str, Any],
    fps: int,
    dpi: int,
    figsize: Tuple[float, float],
    num_b: int,
    num_angles: int,
    num_points: int,
    padding_frac: float,
    max_overlay_tracks: int,
    overlay_lines: bool,
    overlay_line_count: int,
    confidence_min_count: int,
    fixed_energy_range: Optional[Tuple[float, float]],
    hexbin_gridsize: int,
    kde_sigma_bins: float,
    time_frames: int,
) -> list[Path]:
    figs = plot_trajectory_tracks(
        narrowed=ctx["narrowed"],
        sols_best=ctx["sols_best"],
        analyses_best=ctx["analyses_best"],
        cfg=ctx["cfg"],
        num_b=num_b,
        num_angles=num_angles,
        num_points=num_points,
        padding_frac=padding_frac,
        max_overlay_tracks=max_overlay_tracks,
        overlay_lines=overlay_lines,
        overlay_line_count=overlay_line_count,
        gradient_mode="time_video",
        confidence_min_count=confidence_min_count,
        fixed_energy_range=fixed_energy_range,
        hexbin_gridsize=hexbin_gridsize,
        kde_sigma_bins=kde_sigma_bins,
        time_frames=time_frames,
        export_phase_data=True,
        export_time_data=True,
        save_dir=str(output_dir),
        dpi=dpi,
    )
    for fig in figs:
        plt.close(fig)

    star_npz = output_dir / "trajectory_time_data_star.npz"
    planet_npz = output_dir / "trajectory_time_data_planet.npz"

    out_star = make_single_body_gif(
        star_npz,
        output_dir / "trajectory_gradient_evolution_star.gif",
        title="Trajectory Gradient Evolution (Star)",
        fps=fps,
        dpi=dpi,
        figsize=figsize,
    )
    out_planet = make_single_body_gif(
        planet_npz,
        output_dir / "trajectory_gradient_evolution_planet.gif",
        title="Trajectory Gradient Evolution (Planet)",
        fps=fps,
        dpi=dpi,
        figsize=figsize,
    )
    out_combined = make_combined_gif(
        star_npz,
        planet_npz,
        output_dir / "trajectory_gradient_evolution_star_planet.gif",
        fps=fps,
        dpi=dpi,
        figsize=figsize,
    )

    return [out_star, out_planet, out_combined]


def parse_fixed_energy(values: Optional[list[float]]) -> Optional[Tuple[float, float]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--fixed-energy-range expects exactly two values: vmin vmax")
    vmin, vmax = float(values[0]), float(values[1])
    if vmax <= vmin:
        raise ValueError("--fixed-energy-range requires vmax > vmin")
    return (vmin, vmax)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render trajectory gradients from latest record.")
    p.add_argument("--run-dir", type=Path, default=None, help="Specific run directory (contains results.pkl + config.yaml)")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--output-tag", type=str, default=None, help="Optional output subdir name; defaults to run dir name")

    p.add_argument("--mode", choices=["static", "video", "both"], default="both")
    p.add_argument("--static-gradient-mode", choices=["legacy", "line_overlay", "hexbin", "kde"], default="legacy")

    p.add_argument("--static-dpi", type=int, default=180)
    p.add_argument("--video-dpi", type=int, default=180)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--frames", type=int, default=60)

    p.add_argument("--hexbin-gridsize", type=int, default=160)
    p.add_argument("--kde-sigma-bins", type=float, default=2.0)

    p.add_argument("--num-b", type=int, default=160)
    p.add_argument("--num-angles", type=int, default=140)
    p.add_argument("--num-points", type=int, default=260)
    p.add_argument("--padding-frac", type=float, default=0.20)
    p.add_argument("--max-overlay-tracks", type=int, default=260)
    p.add_argument("--overlay-lines", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overlay-line-count", type=int, default=110)
    p.add_argument("--confidence-min-count", type=int, default=2)
    p.add_argument("--fixed-energy-range", nargs=2, type=float, default=None)

    p.add_argument("--fig-width", type=float, default=14.0)
    p.add_argument("--fig-height", type=float, default=8.0)
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    run_dir = args.run_dir if args.run_dir else find_latest_run_dir(args.results_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = resolve_output_dir(args.output_root, run_dir, args.output_tag)
    fixed_energy_range = parse_fixed_energy(args.fixed_energy_range)
    figsize = (float(args.fig_width), float(args.fig_height))

    ctx = load_run_context(run_dir)

    generated: list[Path] = []
    if args.mode in {"static", "both"}:
        generated.extend(
            render_static(
                output_dir=output_dir,
                ctx=ctx,
                gradient_mode=args.static_gradient_mode,
                dpi=int(args.static_dpi),
                figsize=figsize,
                num_b=int(args.num_b),
                num_angles=int(args.num_angles),
                num_points=int(args.num_points),
                padding_frac=float(args.padding_frac),
                max_overlay_tracks=int(args.max_overlay_tracks),
                overlay_lines=bool(args.overlay_lines),
                overlay_line_count=int(args.overlay_line_count),
                confidence_min_count=int(args.confidence_min_count),
                fixed_energy_range=fixed_energy_range,
                hexbin_gridsize=int(args.hexbin_gridsize),
                kde_sigma_bins=float(args.kde_sigma_bins),
                time_frames=int(args.frames),
            )
        )

    if args.mode in {"video", "both"}:
        generated.extend(
            render_video(
                output_dir=output_dir,
                ctx=ctx,
                fps=int(args.fps),
                dpi=int(args.video_dpi),
                figsize=figsize,
                num_b=int(args.num_b),
                num_angles=int(args.num_angles),
                num_points=int(args.num_points),
                padding_frac=float(args.padding_frac),
                max_overlay_tracks=int(args.max_overlay_tracks),
                overlay_lines=bool(args.overlay_lines),
                overlay_line_count=int(args.overlay_line_count),
                confidence_min_count=int(args.confidence_min_count),
                fixed_energy_range=fixed_energy_range,
                hexbin_gridsize=int(args.hexbin_gridsize),
                kde_sigma_bins=float(args.kde_sigma_bins),
                time_frames=int(args.frames),
            )
        )

    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Generated files: {len(generated)}")
    for p in generated:
        print(f" - {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
