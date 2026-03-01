"""
Auto-generated markdown run report.

Produces a ``REPORT.md`` with:
- Scientific-style narrative sections (abstract, methods, results, conclusions)
- Quantitative summary tables for Monte Carlo and top candidates
- Embedded figures with short per-figure summaries
- Baseline comparison and derived amplification metrics
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence
import numpy as np
from datetime import datetime
import html

from ..config import FullConfig
from ..constants import R_SUN, R_JUP


def _fmt_num(v: Any, digits: int = 3, sci: bool = False) -> str:
    """Format numeric values consistently for markdown tables."""
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "n/a"
    if sci:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _fmt_bool(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    return "n/a"


def _html(v: Any) -> str:
    """HTML-escape arbitrary cell content."""
    return html.escape(str(v))


def _finite(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


def _stat_row(arr: np.ndarray) -> Optional[Dict[str, float]]:
    a = _finite(arr)
    if a.size == 0:
        return None
    return {
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _enc_get(enc: Any, key: str, default: Any = None) -> Any:
    """Read encounter field from either dataclass-like object or dict."""
    if enc is None:
        return default
    if isinstance(enc, dict):
        return enc.get(key, default)
    return getattr(enc, key, default)


def _collect_plot_names(saved_plots: Optional[List[str]], output_dir: Path) -> List[str]:
    """Collect unique plot file names relative to ``output_dir``."""
    names: List[str] = []
    seen = set()

    if saved_plots:
        for sp in saved_plots:
            name = Path(sp).name
            if name and name not in seen:
                names.append(name)
                seen.add(name)

    # Include any PNGs in output dir that may not have been tracked in saved_plots
    for p in sorted(output_dir.glob("*.png")):
        name = p.name
        if name not in seen:
            names.append(name)
            seen.add(name)

    return names


def _figure_meta(name: str) -> Dict[str, str]:
    """Title/caption lookup for known diagnostic figures."""
    lookup = {
        "mc_summary.png": {
            "title": "Monte Carlo Outcome Map",
            "caption": "Distribution of trial outcomes in Delta-v / deflection space.",
        },
        "rejection_breakdown.png": {
            "title": "Rejection Taxonomy",
            "caption": "Breakdown of failure modes from physical and numerical filters.",
        },
        "parameter_correlations.png": {
            "title": "Parameter Correlation Matrix",
            "caption": "Cross-correlation of sampled parameters and achieved outcomes.",
        },
        "publication_objectives_dashboard.png": {
            "title": "Publication Objectives Dashboard",
            "caption": "Six-panel evidence view linking mechanism, performance, ceilings, and plausibility.",
        },
        "candidate_ranking_diagnostics.png": {
            "title": "Candidate Ranking Diagnostics",
            "caption": "Ranked comparison of scalar and vector outcomes, mechanism metrics, and closest approaches.",
        },
        "star_proximity_distribution.png": {
            "title": "Star-Proximity Distribution",
            "caption": "Distribution of minimum star approach distances for accepted trajectories.",
        },
        "energy_cdf.png": {
            "title": "Energy CDF",
            "caption": "Cumulative distribution of 0.5*DeltaV_vec^2 for successful samples.",
        },
        "best_candidate.png": {
            "title": "Best Scalar Candidate Trajectory",
            "caption": "Geometry of the trajectory that maximizes scalar Delta-v.",
        },
        "velocity_phase_space.png": {
            "title": "Velocity Phase Space",
            "caption": "Evolution of velocity components for the selected best trajectory.",
        },
        "planet_frame_diagnostics.png": {
            "title": "Planet-Frame Diagnostics",
            "caption": "Relative in/out speeds and planet-frame deflection for top candidates.",
        },
        "multi_candidate_overlay.png": {
            "title": "Top-Candidate Overlay",
            "caption": "Visual comparison of top-ranked candidate trajectories.",
        },
        "trajectory_tracks_star.png": {
            "title": "Narrowed Baseline Tracks (Star)",
            "caption": "2-body narrowed-envelope trajectory tracks for star scattering.",
        },
        "trajectory_tracks_planet.png": {
            "title": "Narrowed Baseline Tracks (Planet)",
            "caption": "2-body narrowed-envelope trajectory tracks for planet scattering.",
        },
        "barycentric_comparison.png": {
            "title": "Barycentric Baseline Comparison",
            "caption": "3-body trajectory compared against barycentric monopole baseline.",
        },
        "planet_frame_comparison.png": {
            "title": "Planet-Frame Baseline Comparison",
            "caption": "3-body encounter compared against equivalent 2-body planet-frame geometry.",
        },
    }
    return lookup.get(
        name,
        {
            "title": name.replace(".png", "").replace("_", " ").title(),
            "caption": "Diagnostic figure generated by the pipeline.",
        },
    )


def generate_run_report(
    output_dir: Path,
    cfg: FullConfig,
    mc: Dict[str, Any],
    analyses_best: List[Optional[Dict]],
    best: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
    narrowed: Optional[Dict[str, Any]] = None,
    saved_plots: Optional[List[str]] = None,
    top_indices: Optional[Sequence[int]] = None,
) -> str:
    """Generate ``REPORT.md`` and return markdown text."""
    lines: List[str] = []
    table_active = False

    def h1(t: str):
        close_table()
        lines.append(f'<h1 align="center">{_html(t)}</h1>')

    def h2(t: str):
        close_table()
        lines.append(f"## {t}")

    def h3(t: str):
        close_table()
        lines.append(f"### {t}")

    def p(t: str):
        close_table()
        lines.append(f"{t}")

    def table_header(*cols: str):
        nonlocal table_active
        close_table()
        lines.append('<div align="center">')
        lines.append(
            '<table style="width: 94%; font-size: 0.90em; border-collapse: collapse; '
            'table-layout: auto;">'
        )
        lines.append("<thead><tr>")
        for c in cols:
            lines.append(
                f'<th style="padding: 6px 8px; text-align: center; '
                f'border-bottom: 1px solid #999;">{_html(c)}</th>'
            )
        lines.append("</tr></thead>")
        lines.append("<tbody>")
        table_active = True

    def table_row(*cols: Any, raw: bool = False):
        nonlocal table_active
        if not table_active:
            return
        lines.append("<tr>")
        for c in cols:
            txt = str(c)
            rendered = txt if raw else _html(txt)
            lines.append(
                f'<td style="padding: 4px 7px; text-align: center; '
                f'border-bottom: 1px solid #e2e2e2;">{rendered}</td>'
            )
        lines.append("</tr>")

    def close_table():
        nonlocal table_active
        if table_active:
            lines.append("</tbody></table>")
            lines.append("</div>")
            lines.append("")
            table_active = False

    R_star_km = cfg.system.R_star_Rsun * R_SUN
    R_p_km = cfg.system.R_planet_Rjup * R_JUP

    ok = np.asarray(mc.get("ok", []), dtype=bool)
    N = int(ok.size)
    ok_count = int(ok.sum()) if N > 0 else 0
    success_pct = (100.0 * ok_count / N) if N > 0 else 0.0

    valid_rerun = [(i, a) for i, a in enumerate(analyses_best) if a is not None]
    valid_count = len(valid_rerun)
    invalid_count = len(analyses_best) - valid_count

    best_ana = best.get("best_ana")
    best_vec_ana = best.get("best_vec_ana")
    best_idx = best.get("best_idx")
    best_vec_idx = best.get("best_vec_idx")
    top_plot_names = _collect_plot_names(saved_plots, output_dir)

    dv_ok = _finite(np.asarray(mc.get("delta_v", np.array([])))[ok]) if N > 0 and "delta_v" in mc else np.array([])
    dv_vec_ok = _finite(np.asarray(mc.get("delta_v_vec", np.array([])))[ok]) if N > 0 and "delta_v_vec" in mc else np.array([])
    defl_ok = _finite(np.asarray(mc.get("deflection", np.array([])))[ok]) if N > 0 and "deflection" in mc else np.array([])

    reasons: Dict[str, int] = {}
    for r in mc.get("results", []):
        if r.get("ok", False):
            reasons["success"] = reasons.get("success", 0) + 1
        else:
            reason = str(r.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
    dominant_reject = None
    reject_only = {k: v for k, v in reasons.items() if k != "success"}
    if reject_only:
        dominant_reject = max(reject_only.items(), key=lambda kv: kv[1])

    # Header
    h1(f"Slingshot Pipeline Report - {cfg.system.name}")
    p(f'<p align="center"><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    p(f'<p align="center"><strong>Output directory:</strong> <code>{_html(output_dir)}</code></p>')
    p("")

    # Abstract
    h2("Abstract")
    abstract_bits: List[str] = []
    abstract_bits.append(
        f"A restricted 3-body Monte Carlo campaign was executed for {cfg.system.name} "
        f"using {N:,} initial conditions in the barycentric frame."
    )
    abstract_bits.append(
        f"The acceptance rate after physical filters was {success_pct:.2f}% "
        f"({ok_count:,}/{N:,} trajectories)."
    )
    if best_ana is not None:
        abstract_bits.append(
            f"The best scalar-speed candidate (MC#{best_idx}) achieved Delta-v={_fmt_num(best_ana.get('delta_v'), 3)} km/s "
            f"with deflection={_fmt_num(best_ana.get('deflection'), 2)} deg."
        )
    if best_vec_ana is not None:
        abstract_bits.append(
            f"The strongest vector-turning candidate (MC#{best_vec_idx}) achieved "
            f"DeltaV_vec={_fmt_num(best_vec_ana.get('delta_v_vec'), 3)} km/s "
            f"(0.5*DeltaV^2={_fmt_num(best_vec_ana.get('energy_half_dv_vec_sq'), 3)} km^2/s^2)."
        )
    if dominant_reject is not None:
        abstract_bits.append(
            f"The dominant rejection mode was `{dominant_reject[0]}` "
            f"({dominant_reject[1]:,} trajectories)."
        )
    p(" ".join(abstract_bits))
    p("")

    # Executive summary
    h2("Executive Summary")
    table_header("Metric", "Value")
    table_row("Particles (MC)", f"{N:,}")
    table_row("Successful trajectories", f"{ok_count:,} ({success_pct:.1f}%)")
    table_row("Re-run candidates (total)", f"{len(analyses_best):,}")
    table_row("Re-run candidates (valid)", f"{valid_count:,}")
    table_row("Re-run candidates (invalid)", f"{invalid_count:,}")
    if top_indices is not None:
        table_row("Selected candidate count", f"{len(top_indices):,}")
    if best_ana is not None:
        table_row(
            "Best scalar candidate",
            f"MC#{best_idx}  Delta v={_fmt_num(best_ana.get('delta_v'), 4)} km/s "
            f"({_fmt_num(best_ana.get('delta_v_pct'), 2)}%)",
        )
    if best_vec_ana is not None:
        table_row(
            "Best vector candidate",
            f"MC#{best_vec_idx}  DeltaV_vec_mag={_fmt_num(best_vec_ana.get('delta_v_vec'), 4)} km/s, "
            f"0.5*DeltaV^2={_fmt_num(best_vec_ana.get('energy_half_dv_vec_sq'), 4)} km^2/s^2",
        )
    p("")

    h3("Key Findings")
    if dv_ok.size > 0:
        p(f"- Scalar Delta-v range across accepted trajectories: [{dv_ok.min():.3f}, {dv_ok.max():.3f}] km/s.")
        p(f"- Scalar Delta-v mean +- std: {dv_ok.mean():.3f} +- {dv_ok.std():.3f} km/s.")
    if dv_vec_ok.size > 0:
        p(f"- Vector-change magnitude range: [{dv_vec_ok.min():.3f}, {dv_vec_ok.max():.3f}] km/s.")
    if defl_ok.size > 0:
        p(f"- Deflection range: [{defl_ok.min():.2f}, {defl_ok.max():.2f}] deg.")
    if best_ana is not None and best_vec_ana is not None:
        p(
            f"- Best-scalar (MC#{best_idx}) and best-vector (MC#{best_vec_idx}) "
            f"are {'the same' if best_idx == best_vec_idx else 'different'} trajectories."
        )
    if dominant_reject is not None:
        pct = (100.0 * dominant_reject[1] / N) if N > 0 else 0.0
        p(f"- Dominant rejection class: `{dominant_reject[0]}` ({dominant_reject[1]:,}, {pct:.1f}%).")
    p("")

    # System configuration
    h2("System Configuration")
    table_header("Parameter", "Value")
    table_row("Star mass", f"{_fmt_num(cfg.system.M_star_Msun, 3)} M_sun")
    table_row("Star radius", f"{_fmt_num(cfg.system.R_star_Rsun, 3)} R_sun ({_fmt_num(R_star_km, 0)} km)")
    table_row("Planet mass", f"{_fmt_num(cfg.system.M_planet_Mjup, 3)} M_jup")
    table_row("Planet radius", f"{_fmt_num(cfg.system.R_planet_Rjup, 3)} R_jup ({_fmt_num(R_p_km, 0)} km)")
    table_row("Semi-major axis", f"{_fmt_num(cfg.system.a_planet_AU, 5)} AU")
    vb_x = cfg.system.bulk_velocity_vx_kms
    vb_y = cfg.system.bulk_velocity_vy_kms
    table_row("System bulk velocity", f"({vb_x:.3f}, {vb_y:.3f}) km/s  speed={np.hypot(vb_x, vb_y):.3f} km/s")
    p("")

    # Numerical + sampling configuration
    h2("Methods and Configuration")
    h3("Sampling and Numerical Setup")
    table_header("Parameter", "Value")
    table_row("Sampling mode", cfg.sampling.mode)
    table_row(
        "Velocity range",
        f"[{_fmt_num(cfg.sampling.v_mag_min_kms, 2)}, {_fmt_num(cfg.sampling.v_mag_max_kms, 2)}] km/s",
    )
    table_row(
        "Impact parameter range",
        f"[{_fmt_num(cfg.sampling.impact_param_min_AU, 4)}, {_fmt_num(cfg.sampling.impact_param_max_AU, 4)}] AU",
    )
    table_row(
        "Incoming angle range",
        f"[{_fmt_num(cfg.sampling.angle_in_min_deg, 1)}, {_fmt_num(cfg.sampling.angle_in_max_deg, 1)}] deg",
    )
    table_row("Barycentric unbound requirement", str(cfg.sampling.bary_unbound_requirement))
    table_row("ODE method", cfg.numerical.ode_method)
    table_row("rtol / atol", f"{cfg.numerical.rtol} / {cfg.numerical.atol}")
    table_row("r_far_factor", _fmt_num(cfg.numerical.r_far_factor, 2))
    table_row("escape_radius_factor", _fmt_num(cfg.numerical.escape_radius_factor, 2))
    table_row("softening", f"{_fmt_num(cfg.numerical.softening_km, 2)} km")
    table_row("flyby_r_min_max_hill", str(cfg.numerical.flyby_r_min_max_hill))
    table_row("star_min_clearance_Rstar", str(cfg.numerical.star_min_clearance_Rstar))
    p("")

    # Selection configuration
    h3("Candidate Selection Setup")
    table_header("Parameter", "Value")
    table_row("select_mode", cfg.pipeline.select_mode)
    table_row("top_frac / min_top", f"{cfg.pipeline.top_frac} / {cfg.pipeline.min_top}")
    if cfg.pipeline.select_mode == "single":
        table_row("metric / sign", f"{cfg.pipeline.select_metric} / {cfg.pipeline.select_sign}")
    else:
        table_row("weighted_normalization", cfg.pipeline.weighted_normalization)
        p("")
        table_header("Objective metric", "Sign", "Weight")
        for obj in cfg.pipeline.selection_objectives:
            table_row(obj.metric, obj.sign, _fmt_num(obj.weight, 3))
    p("")

    # Results
    h2("Results")
    h3("Monte Carlo Summary Statistics")
    metric_specs = [
        ("delta_v", "Delta v", "km/s"),
        ("delta_v_pct", "Delta v percent", "%"),
        ("delta_v_vec", "DeltaV_vec magnitude", "km/s"),
        ("energy_half_dv_vec_sq", "0.5*DeltaV_vec^2", "km^2/s^2"),
        ("deflection", "Deflection", "deg"),
        ("r_min", "r_min planet", "km"),
        ("r_star_min", "r_min star", "km"),
        ("delta_v_planet_frame", "Delta v planet frame", "km/s"),
        ("energy_from_planet_orbit", "Energy from planet orbit", "km^2/s^2"),
    ]
    table_header("Metric", "Min", "P05", "Median", "P95", "Max", "Mean +- std")
    for key, label, unit in metric_specs:
        if key not in mc:
            continue
        stats = _stat_row(np.asarray(mc[key])[ok]) if N > 0 else None
        if stats is None:
            continue
        table_row(
            f"{label} ({unit})",
            _fmt_num(stats["min"], 3),
            _fmt_num(stats["p05"], 3),
            _fmt_num(stats["p50"], 3),
            _fmt_num(stats["p95"], 3),
            _fmt_num(stats["max"], 3),
            f"{_fmt_num(stats['mean'], 3)} +- {_fmt_num(stats['std'], 3)}",
        )
    p("")

    # Rejection breakdown
    h3("Rejection Breakdown")
    table_header("Reason", "Count", "Percent")
    for reason in sorted(reasons, key=reasons.get, reverse=True):
        cnt = reasons[reason]
        pct = (100.0 * cnt / N) if N > 0 else 0.0
        table_row(reason, cnt, f"{pct:.1f}%")
    p("")

    # Candidate summary table (notebook-like)
    h3("Top Candidates Summary")
    ranked: List[tuple[int, Dict[str, Any]]] = []
    if valid_count == 0:
        p("No valid re-run candidates were available.")
        p("")
    else:
        # Keep deterministic ordering by scalar Delta v (descending).
        ranked = sorted(
            valid_rerun,
            key=lambda x: float(x[1].get("delta_v", -np.inf)),
            reverse=True,
        )
        table_header(
            "Rank",
            "MC_idx",
            "Delta v (km/s)",
            "Delta v (%)",
            "Deflection (deg)",
            "r_min_planet (km)",
            "r_min_star (km)",
            "impact_param (km)",
            "Tag",
        )
        for rank, (i, ana) in enumerate(ranked, start=1):
            mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
            enc = ana.get("encounter")
            r_star = _enc_get(enc, "r_star_min", np.nan)
            tag = []
            if best_idx is not None and mc_idx == int(best_idx):
                tag.append("best_scalar")
            if best_vec_idx is not None and mc_idx == int(best_vec_idx):
                tag.append("best_vector")
            table_row(
                rank,
                mc_idx,
                _fmt_num(ana.get("delta_v"), 3),
                _fmt_num(ana.get("delta_v_pct"), 2),
                _fmt_num(ana.get("deflection"), 1),
                _fmt_num(ana.get("r_min"), 0),
                _fmt_num(r_star, 0),
                _fmt_num(ana.get("impact_parameter"), 0),
                ", ".join(tag) if tag else "-",
            )
        p("")

        h3("Planet-Frame Diagnostics (Top Candidates)")
        table_header(
            "MC_idx",
            "v_rel_in (km/s)",
            "v_rel_out (km/s)",
            "Delta v planet frame (km/s)",
            "planet deflection (deg)",
            "energy from planet orbit (km^2/s^2)",
            "Delta eps monopole (km^2/s^2)",
        )
        for i, ana in ranked:
            mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
            table_row(
                mc_idx,
                _fmt_num(ana.get("v_rel_planet_in"), 3),
                _fmt_num(ana.get("v_rel_planet_out"), 3),
                _fmt_num(ana.get("delta_v_planet_frame"), 3),
                _fmt_num(ana.get("planet_deflection_deg"), 2),
                _fmt_num(ana.get("energy_from_planet_orbit"), 6, sci=True),
                _fmt_num(ana.get("delta_eps_monopole"), 6, sci=True),
            )
        p("")

    def emit_best_block(title: str, mc_idx: Any, ana: Optional[Dict[str, Any]]):
        h3(title)
        if ana is None:
            p("No candidate available.")
            p("")
            return

        enc = ana.get("encounter")
        r_min = ana.get("r_min")
        r_star = _enc_get(enc, "r_star_min", np.nan)
        star_v = _enc_get(enc, "star_v_bary_in", None)
        star_v_str = "n/a"
        if star_v is not None and len(star_v) >= 2:
            svx = float(star_v[0])
            svy = float(star_v[1])
            star_v_str = f"({svx:.4f}, {svy:.4f}) km/s, speed={np.hypot(svx, svy):.4f} km/s"

        table_header("Metric", "Value")
        table_row("MC index", mc_idx)
        table_row("v_i / v_f", f"{_fmt_num(ana.get('v_i'), 4)} / {_fmt_num(ana.get('v_f'), 4)} km/s")
        table_row("Delta v (scalar)", f"{_fmt_num(ana.get('delta_v'), 4)} km/s ({_fmt_num(ana.get('delta_v_pct'), 2)}%)")
        table_row("DeltaV_vec magnitude", f"{_fmt_num(ana.get('delta_v_vec'), 4)} km/s")
        table_row("0.5*DeltaV_vec^2", f"{_fmt_num(ana.get('energy_half_dv_vec_sq'), 4)} km^2/s^2")
        table_row("Deflection", f"{_fmt_num(ana.get('deflection'), 3)} deg")
        table_row("Impact parameter", f"{_fmt_num(ana.get('impact_parameter'), 3)} km")
        table_row("r_min (planet)", f"{_fmt_num(r_min, 0)} km ({_fmt_num((float(r_min) / R_p_km) if r_min else np.nan, 2)} R_p)")
        table_row("r_min (star)", f"{_fmt_num(r_star, 0)} km ({_fmt_num((float(r_star) / R_star_km) if np.isfinite(r_star) else np.nan, 2)} R_star)")
        table_row("unbound_i / unbound_f", f"{_fmt_bool(ana.get('unbound_i'))} / {_fmt_bool(ana.get('unbound_f'))}")
        table_row("eps_i / eps_f", f"{_fmt_num(ana.get('eps_i'), 6, sci=True)} / {_fmt_num(ana.get('eps_f'), 6, sci=True)} km^2/s^2")
        table_row("v_rel planet in / out", f"{_fmt_num(ana.get('v_rel_planet_in'), 4)} / {_fmt_num(ana.get('v_rel_planet_out'), 4)} km/s")
        table_row("Delta v planet frame", f"{_fmt_num(ana.get('delta_v_planet_frame'), 4)} km/s")
        table_row("Planet deflection", f"{_fmt_num(ana.get('planet_deflection_deg'), 3)} deg")
        table_row("Energy from planet orbit", f"{_fmt_num(ana.get('energy_from_planet_orbit'), 6, sci=True)} km^2/s^2")
        table_row("Delta eps monopole", f"{_fmt_num(ana.get('delta_eps_monopole'), 6, sci=True)} km^2/s^2")
        table_row("Star bary velocity at encounter", star_v_str)
        p("")

    h3("Best Candidate Diagnostics")
    emit_best_block("Best by Scalar Delta v", best_idx, best_ana)
    emit_best_block("Best by Vector Metric (0.5|Delta V|^2)", best_vec_idx, best_vec_ana)

    # Baselines and comparison
    h3("2-Body vs 3-Body Comparison")
    if comparison is None:
        p("No comparison data available.")
        p("")
    else:
        table_header("Quantity", "Value")
        table_row("baseline_mode", comparison.get("baseline_mode", "n/a"))
        table_row("unit_energy", comparison.get("unit_energy", "n/a"))
        table_row("unit_dv", comparison.get("unit_dv", "n/a"))
        for key in [
            "energy_2body_star",
            "energy_2body_planet",
            "energy_3body",
            "delta_E_vs_star",
            "pct_vs_star",
            "delta_E_vs_planet",
            "pct_vs_planet",
            "dv_2body_star",
            "dv_2body_planet",
            "dv_3body",
            "dv_improvement_vs_star",
            "dv_improvement_vs_planet",
            "dv_vec_2body_star",
            "dv_vec_2body_planet",
            "dv_vec_3body",
            "dv_vec_improvement_vs_star",
            "dv_vec_pct_vs_star",
            "dv_vec_improvement_vs_planet",
            "dv_vec_pct_vs_planet",
        ]:
            if key in comparison:
                table_row(key, _fmt_num(comparison[key], 6))
        p("")

        # Planet-ceiling amplification metric, matching notebook interpretation.
        if "dv_vec_3body" in comparison and "dv_vec_2body_planet" in comparison:
            e3 = 0.5 * float(comparison["dv_vec_3body"]) ** 2
            ep = 0.5 * float(comparison["dv_vec_2body_planet"]) ** 2
            if ep > 0:
                amp = e3 / ep
                p(f"Planet slingshot amplification (energy ratio): **{amp:.3f}x**")
                p(f"(3-body 0.5*DeltaV^2 = {e3:.3f} vs planet 2-body = {ep:.3f} km^2/s^2)")
                p("")

        env_txt = comparison.get("envelope_summary")
        if env_txt:
            h3("Narrowed Envelope")
            p("```text")
            p(env_txt)
            p("```")
            p("")

    # Narrowed baseline sweep stats
    if narrowed:
        h3("Narrowed Baseline Sweep Statistics")
        table_header(
            "Body",
            "Valid / Total",
            "Max epsilon (km^2/s^2)",
            "Max |Delta V_vec| (km/s)",
            "Max 0.5|Delta V|^2 (km^2/s^2)",
        )
        for body in ["star", "planet"]:
            item = narrowed.get(body)
            if item is None:
                continue
            table_row(
                body,
                f"{item.n_valid}/{item.n_total}",
                _fmt_num(item.max_epsilon, 4),
                _fmt_num(item.max_deltaV_vec, 4),
                _fmt_num(item.max_energy_half_dv_vec_sq, 4),
            )
        p("")

    # Figure gallery
    h2("Figure Gallery")
    if not top_plot_names:
        p("No figures were available for embedding.")
        p("")
    else:
        preferred = [
            "mc_summary.png",
            "rejection_breakdown.png",
            "parameter_correlations.png",
            "publication_objectives_dashboard.png",
            "candidate_ranking_diagnostics.png",
            "star_proximity_distribution.png",
            "energy_cdf.png",
            "best_candidate.png",
            "velocity_phase_space.png",
            "planet_frame_diagnostics.png",
            "multi_candidate_overlay.png",
            "trajectory_tracks_star.png",
            "trajectory_tracks_planet.png",
            "barycentric_comparison.png",
            "planet_frame_comparison.png",
        ]
        ordered = [n for n in preferred if n in top_plot_names] + [n for n in top_plot_names if n not in preferred]

        def _figure_summary(name: str) -> str:
            if name == "mc_summary.png":
                if dv_ok.size > 0:
                    return (
                        f"Accepted solutions: {ok_count:,}/{N:,} ({success_pct:.2f}%). "
                        f"Scalar Delta-v spans [{dv_ok.min():.3f}, {dv_ok.max():.3f}] km/s."
                    )
                return f"Accepted solutions: {ok_count:,}/{N:,} ({success_pct:.2f}%)."
            if name == "rejection_breakdown.png" and dominant_reject is not None:
                pct = (100.0 * dominant_reject[1] / N) if N > 0 else 0.0
                return f"Dominant rejection: `{dominant_reject[0]}` ({dominant_reject[1]:,}, {pct:.1f}%)."
            if name == "publication_objectives_dashboard.png" and comparison is not None:
                dv3 = comparison.get("dv_vec_3body")
                dvp = comparison.get("dv_vec_2body_planet")
                if dv3 is not None and dvp is not None and float(dvp) != 0.0:
                    ratio = (0.5 * float(dv3) ** 2) / (0.5 * float(dvp) ** 2)
                    return f"Dashboard ties objectives to evidence; 3-body vs planet-only energy ratio is {ratio:.2f}x."
                return "Dashboard links candidate outcomes to mechanism and baseline ceilings."
            if name == "candidate_ranking_diagnostics.png" and ranked:
                top_lbl = int(top_indices[ranked[0][0]]) if top_indices is not None else ranked[0][0]
                return f"Ranking view highlights tradeoffs across top {len(ranked)} re-run candidates (highest scalar: MC#{top_lbl})."
            if name == "energy_cdf.png" and dv_vec_ok.size > 0:
                e_med = 0.5 * float(np.median(dv_vec_ok)) ** 2
                return f"Median accepted scattering energy: 0.5*DeltaV_vec^2 = {e_med:.3f} km^2/s^2."
            if name == "best_candidate.png" and best_ana is not None:
                return (
                    f"Best scalar candidate MC#{best_idx}: Delta-v={_fmt_num(best_ana.get('delta_v'), 3)} km/s, "
                    f"deflection={_fmt_num(best_ana.get('deflection'), 2)} deg."
                )
            if name == "planet_frame_diagnostics.png" and ranked:
                vals = np.array([float(a.get("delta_v_planet_frame", np.nan)) for _, a in ranked], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    return f"Median Delta-v in planet frame across top candidates: {np.median(vals):.3f} km/s."
            if name == "multi_candidate_overlay.png" and ranked:
                return f"Overlay includes top {len(ranked)} valid re-run trajectories."
            if name in {"trajectory_tracks_star.png", "trajectory_tracks_planet.png"} and comparison:
                mode = comparison.get("baseline_mode", "n/a")
                return f"Tracks correspond to narrowed-envelope 2-body baselines (mode={mode})."
            return "See caption for diagnostic context."

        fig_idx = 1
        for name in ordered:
            meta = _figure_meta(name)
            h3(f"Figure {fig_idx}. {meta['title']}")
            p(
                f'<p align="center">'
                f'<img src="{_html(name)}" alt="{_html(f"Figure {fig_idx}: {meta["title"]}")}" '
                f'style="max-width: 86%; height: auto;" />'
                f'</p>'
            )
            p(
                f'<p align="center"><em>Figure {fig_idx}. {meta["title"]}. '
                f'{meta["caption"]}</em></p>'
            )
            p(f'<p align="center"><em>Summary:</em> {_html(_figure_summary(name))}</p>')
            p("")
            fig_idx += 1

    # Plot index
    if top_plot_names:
        h2("Figure Index")
        table_header("Figure file", "Relative path")
        for name in top_plot_names:
            table_row(f'<a href="{_html(name)}">{_html(name)}</a>', _html(name), raw=True)
        p("")

    # Closing summary
    h2("Conclusions")
    if best_ana is None and best_vec_ana is None:
        p("No valid candidate could be promoted to the final diagnostics stage.")
    else:
        if best_ana is not None:
            p(
                f"- Best scalar-speed outcome: MC#{best_idx}, Delta-v={_fmt_num(best_ana.get('delta_v'), 4)} km/s "
                f"({_fmt_num(best_ana.get('delta_v_pct'), 2)}%)."
            )
        if best_vec_ana is not None:
            p(
                f"- Best vector-turn outcome: MC#{best_vec_idx}, DeltaV_vec={_fmt_num(best_vec_ana.get('delta_v_vec'), 4)} km/s, "
                f"0.5*DeltaV^2={_fmt_num(best_vec_ana.get('energy_half_dv_vec_sq'), 4)} km^2/s^2."
            )
        if comparison is not None and "dv_vec_3body" in comparison and "dv_vec_2body_planet" in comparison:
            e3 = 0.5 * float(comparison["dv_vec_3body"]) ** 2
            ep = 0.5 * float(comparison["dv_vec_2body_planet"]) ** 2
            if ep > 0:
                p(f"- Relative to the planet-only 2-body ceiling, the 3-body best reaches an energy ratio of {e3/ep:.3f}x.")
        if dominant_reject is not None:
            p(f"- Throughput is primarily limited by `{dominant_reject[0]}` in the current sampling envelope.")
    p("")

    # Footer
    p("---")
    from . import __version__
    p(f"*slingshot-solver v{__version__}*")

    close_table()
    report_text = "\n".join(lines)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text
