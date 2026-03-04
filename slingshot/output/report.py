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
    uncertainty_results: Optional[Dict[str, Any]] = None,
    robustness_results: Optional[Dict[str, Any]] = None,
    tiered: Optional[Dict[str, List]] = None,
    tier_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Generate ``REPORT.md`` and return markdown text."""
    lines: List[str] = []
    table_active = False

    # ── CSS style block (injected once at top) ──
    _CSS = """<style>
.ss-table {width: 94%; font-size: 0.90em; border-collapse: collapse; table-layout: auto; margin: 0 auto;}
.ss-table th {padding: 6px 8px; text-align: center; border-bottom: 2px solid #555; background: #f7f7f7;}
.ss-table td {padding: 4px 7px; text-align: center; border-bottom: 1px solid #e2e2e2;}
.ss-table tr:hover td {background: #f0f6ff;}
.ss-tier-planet {border-left: 4px solid #2e7d32;}
.ss-tier-hybrid {border-left: 4px solid #e65100;}
.ss-tier-star   {border-left: 4px solid #1565c0;}
.ss-tier-badge {display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: 600;}
.ss-tier-badge.planet {background: #c8e6c9; color: #1b5e20;}
.ss-tier-badge.hybrid {background: #ffe0b2; color: #bf360c;}
.ss-tier-badge.star   {background: #bbdefb; color: #0d47a1;}
.ss-toc {columns: 3; column-gap: 24px; list-style: none; padding: 0; font-size: 0.92em;}
.ss-toc li {break-inside: avoid; margin-bottom: 3px;}
.ss-fig-group {margin-top: 20px; padding-top: 10px; border-top: 2px solid #ddd;}
</style>
"""

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

    def table_header(*cols: str, css_class: str = ""):
        nonlocal table_active
        close_table()
        lines.append('<div align="center">')
        cls = f"ss-table {css_class}".strip()
        lines.append(f'<table class="{cls}">')
        lines.append("<thead><tr>")
        for c in cols:
            lines.append(f"<th>{_html(c)}</th>")
        lines.append("</tr></thead>")
        lines.append("<tbody>")
        table_active = True

    def table_row(*cols: Any, raw: bool = False, row_class: str = ""):
        nonlocal table_active
        if not table_active:
            return
        cls_attr = f' class="{row_class}"' if row_class else ""
        lines.append(f"<tr{cls_attr}>")
        for c in cols:
            txt = str(c)
            rendered = txt if raw else _html(txt)
            lines.append(f"<td>{rendered}</td>")
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
    lines.append(_CSS)
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

    # Candidate summary table — tiered or flat
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

        # ── Tier breakdown summary ──
        top_n = cfg.tiering.top_n_per_tier if cfg.tiering.enabled else len(ranked)
        if tiered is not None and tier_stats is not None:
            from ..analysis.tiering import TIER_ORDER, TIER_PLANET, TIER_HYBRID, TIER_STAR
            _tier_css = {
                TIER_PLANET: ("ss-tier-planet", "planet"),
                TIER_HYBRID: ("ss-tier-hybrid", "hybrid"),
                TIER_STAR:   ("ss-tier-star",   "star"),
            }

            h3("Tier Classification Summary")
            p(
                "Candidates are classified by ε<sub>planet</sub> / |Δε<sub>monopole</sub>| ratio: "
                f'<span class="ss-tier-badge planet">planet-dominated</span> (&gt;{cfg.tiering.planet_dominated_threshold}), '
                f'<span class="ss-tier-badge hybrid">hybrid</span> ({cfg.tiering.hybrid_threshold}–{cfg.tiering.planet_dominated_threshold}), '
                f'<span class="ss-tier-badge star">star-dominated</span> (&lt;{cfg.tiering.hybrid_threshold}).'
            )
            p("")
            table_header("Tier", "Count", "Δv min (km/s)", "Δv median (km/s)", "Δv max (km/s)", "Ratio range", "Top MC#")
            for tier in TIER_ORDER:
                ts = tier_stats.get(tier, {})
                n = ts.get("count", 0)
                if n == 0:
                    table_row(tier, 0, "-", "-", "-", "-", "-")
                else:
                    rng = f"{_fmt_num(ts.get('ratio_min'), 2)}–{_fmt_num(ts.get('ratio_max'), 2)}"
                    table_row(
                        f'<span class="ss-tier-badge {_tier_css[tier][1]}">{tier}</span>',
                        n,
                        _fmt_num(ts.get("dv_min"), 3),
                        _fmt_num(ts.get("dv_median"), 3),
                        _fmt_num(ts.get("dv_max"), 3),
                        rng,
                        f"MC#{ts.get('top_mc_idx', '?')}",
                        raw=True,
                    )
            p("")

            # ── Per-tier candidate tables (top N per tier) ──
            for tier in TIER_ORDER:
                items = tiered.get(tier, [])
                if not items:
                    continue
                css_row, badge = _tier_css[tier]
                shown = items[:top_n]
                omitted = len(items) - len(shown)
                h3(f"{tier.replace('-', ' ').title()} Candidates (top {len(shown)}" +
                   (f" of {len(items)}" if omitted > 0 else "") + ")")
                table_header(
                    "Rank", "MC#", "Δv (km/s)", "Δv (%)", "Deflection (°)",
                    "r_min planet (km)", "r_min star (km)", "ε/Δε ratio", "Tag",
                    css_class=css_row,
                )
                for rank, mc_idx, ana in shown:
                    enc = ana.get("encounter")
                    r_star = _enc_get(enc, "r_star_min", np.nan)
                    tag = []
                    if best_idx is not None and mc_idx == int(best_idx):
                        tag.append("best_scalar")
                    if best_vec_idx is not None and mc_idx == int(best_vec_idx):
                        tag.append("best_vector")
                    ratio_val = ana.get("tier_ratio")
                    table_row(
                        rank, mc_idx,
                        _fmt_num(ana.get("delta_v"), 3),
                        _fmt_num(ana.get("delta_v_pct"), 2),
                        _fmt_num(ana.get("deflection"), 1),
                        _fmt_num(ana.get("r_min"), 0),
                        _fmt_num(r_star, 0),
                        _fmt_num(ratio_val, 3) if ratio_val is not None else "n/a",
                        ", ".join(tag) if tag else "-",
                    )
                if omitted > 0:
                    p(f"<em>({omitted} additional {tier} candidates omitted — see summary.csv)</em>")
                p("")

        else:
            # Fallback flat table when tiering is disabled
            table_header(
                "Rank", "MC_idx", "Delta v (km/s)", "Delta v (%)",
                "Deflection (deg)", "r_min_planet (km)", "r_min_star (km)",
                "impact_param (km)", "Tag",
            )
            for rank, (i, ana) in enumerate(ranked[:top_n], start=1):
                mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
                enc = ana.get("encounter")
                r_star = _enc_get(enc, "r_star_min", np.nan)
                tag = []
                if best_idx is not None and mc_idx == int(best_idx):
                    tag.append("best_scalar")
                if best_vec_idx is not None and mc_idx == int(best_vec_idx):
                    tag.append("best_vector")
                table_row(
                    rank, mc_idx,
                    _fmt_num(ana.get("delta_v"), 3),
                    _fmt_num(ana.get("delta_v_pct"), 2),
                    _fmt_num(ana.get("deflection"), 1),
                    _fmt_num(ana.get("r_min"), 0),
                    _fmt_num(r_star, 0),
                    _fmt_num(ana.get("impact_parameter"), 0),
                    ", ".join(tag) if tag else "-",
                )
            if len(ranked) > top_n:
                p(f"<em>({len(ranked) - top_n} additional candidates omitted — see summary.csv)</em>")
            p("")

        # ── Planet-Frame Diagnostics (top N only) ──
        h3("Planet-Frame Diagnostics (Top Candidates)")
        pf_show = ranked[:top_n]
        table_header(
            "MC#", "Tier",
            "v_rel_in (km/s)", "v_rel_out (km/s)",
            "Δv planet frame (km/s)", "planet deflection (°)",
            "ε_planet (km²/s²)", "Δε_monopole (km²/s²)",
        )
        for i, ana in pf_show:
            mc_idx = int(top_indices[i]) if top_indices is not None and i < len(top_indices) else i
            tier_label = ana.get("tier", "?")
            table_row(
                mc_idx, tier_label,
                _fmt_num(ana.get("v_rel_planet_in"), 3),
                _fmt_num(ana.get("v_rel_planet_out"), 3),
                _fmt_num(ana.get("delta_v_planet_frame"), 3),
                _fmt_num(ana.get("planet_deflection_deg"), 2),
                _fmt_num(ana.get("energy_from_planet_orbit"), 6, sci=True),
                _fmt_num(ana.get("delta_eps_monopole"), 6, sci=True),
            )
        if len(ranked) > top_n:
            p(f"<em>({len(ranked) - len(pf_show)} additional rows omitted)</em>")
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

    # Figure gallery — section-grouped with TOC
    h2("Figure Gallery")
    if not top_plot_names:
        p("No figures were available for embedding.")
        p("")
    else:
        # ── Figure grouping ──
        _DIAG_NAMES = {
            "mc_summary.png", "rejection_breakdown.png", "parameter_correlations.png",
            "star_proximity_distribution.png", "energy_cdf.png",
        }
        _PUB_NAMES = {
            "publication_objectives_dashboard.png", "candidate_ranking_diagnostics.png",
            "best_candidate.png", "velocity_phase_space.png",
            "planet_frame_diagnostics.png", "multi_candidate_overlay.png",
            "pareto_front_2d.png", "scalar_vs_vector_tradeoff.png",
        }
        _SUPP_NAMES = {
            "trajectory_tracks_star.png", "trajectory_tracks_planet.png",
            "barycentric_comparison.png", "planet_frame_comparison.png",
        }

        def _classify_fig(name: str) -> str:
            if name in _DIAG_NAMES:
                return "diagnostic"
            if name in _PUB_NAMES:
                return "publication"
            if name in _SUPP_NAMES:
                return "supplementary"
            # Heuristic: poincare/heatmap/oberth → supplementary,
            # uncertainty/robustness → supplementary
            lname = name.lower()
            if any(kw in lname for kw in ("poincare", "heatmap", "oberth", "convergence", "sensitivity", "tornado", "uncertainty")):
                return "supplementary"
            if any(kw in lname for kw in ("mc_", "rejection", "star_prox", "parameter_")):
                return "diagnostic"
            return "supplementary"

        preferred = [
            "mc_summary.png", "rejection_breakdown.png", "parameter_correlations.png",
            "star_proximity_distribution.png", "energy_cdf.png",
            "publication_objectives_dashboard.png", "candidate_ranking_diagnostics.png",
            "best_candidate.png", "velocity_phase_space.png",
            "planet_frame_diagnostics.png", "multi_candidate_overlay.png",
            "pareto_front_2d.png", "scalar_vs_vector_tradeoff.png",
            "trajectory_tracks_star.png", "trajectory_tracks_planet.png",
            "barycentric_comparison.png", "planet_frame_comparison.png",
        ]
        ordered = [n for n in preferred if n in top_plot_names] + [n for n in top_plot_names if n not in preferred]

        groups = {"diagnostic": [], "publication": [], "supplementary": []}
        for name in ordered:
            groups[_classify_fig(name)].append(name)

        # TOC
        p("<strong>Contents:</strong>")
        p('<ul class="ss-toc">')
        fig_idx = 1
        toc_map: Dict[str, int] = {}
        for grp_key, grp_label in [("diagnostic", "Diagnostic"), ("publication", "Publication-Quality"), ("supplementary", "Supplementary")]:
            for name in groups[grp_key]:
                meta = _figure_meta(name)
                anchor = f"fig-{fig_idx}"
                toc_map[name] = fig_idx
                p(f'<li><a href="#{anchor}">Fig {fig_idx}. {meta["title"]}</a> <small>({grp_label})</small></li>')
                fig_idx += 1
        p("</ul>")
        p("")

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

        # Emit grouped figures
        for grp_key, grp_label in [("diagnostic", "Diagnostic Figures"), ("publication", "Publication-Quality Figures"), ("supplementary", "Supplementary Figures")]:
            figs_in_group = groups[grp_key]
            if not figs_in_group:
                continue
            p(f'<div class="ss-fig-group">')
            h3(grp_label)
            for name in figs_in_group:
                fi = toc_map[name]
                meta = _figure_meta(name)
                p(f'<a id="fig-{fi}"></a>')
                h3(f"Figure {fi}. {meta['title']}")
                p(
                    f'<p align="center">'
                    f'<img src="{_html(name)}" alt="{_html(f"Figure {fi}: {meta["title"]}")}" '
                    f'style="max-width: 86%; height: auto;" />'
                    f'</p>'
                )
                p(
                    f'<p align="center"><em>Figure {fi}. {meta["title"]}. '
                    f'{meta["caption"]}</em></p>'
                )
                p(f'<p align="center"><em>Summary:</em> {_html(_figure_summary(name))}</p>')
                p("")
            p("</div>")
            p("")

    # Figure index table
    if top_plot_names:
        h2("Figure Index")
        table_header("Fig #", "Group", "File", "Title")
        for name in ordered:
            fi = toc_map.get(name, 0)
            grp = _classify_fig(name)
            meta = _figure_meta(name)
            table_row(
                fi,
                grp,
                f'<a href="#{f"fig-{fi}"}">{_html(name)}</a>',
                meta["title"],
                raw=True,
            )
        p("")

    # ── Pareto front analysis ──
    if cfg.pipeline.select_mode == "pareto" and top_indices is not None and len(valid_rerun) > 0:
        h2("Pareto Front Analysis")
        n_front = len(top_indices)
        p(f"Selection mode: **pareto** — {n_front} non-dominated candidates selected.")
        obj_txt = ", ".join(
            f"{o.metric} ({o.sign})" for o in cfg.pipeline.selection_objectives
        )
        p(f"Objectives: {obj_txt}")
        p("")

    # ── Uncertainty analysis ──
    if uncertainty_results is not None:
        h2("Uncertainty Analysis")

        ci_df = uncertainty_results.get("confidence_bands")
        if ci_df is not None and not ci_df.empty:
            h3("Confidence Intervals (Parameter Posteriors)")
            # Merge tier info if available
            _tier_lookup: Dict[int, str] = {}
            if tiered is not None:
                for tier_label, items in tiered.items():
                    for _, mc_idx, _ in items:
                        _tier_lookup[mc_idx] = tier_label
            ci_cols = [c for c in ci_df.columns if c.endswith("_median") or c.endswith("_lo68") or c.endswith("_hi68") or c.endswith("_lo95") or c.endswith("_hi95")]
            has_tiers = bool(_tier_lookup)
            header_cols = ["candidate"] + (["tier"] if has_tiers else []) + ci_cols[:12]
            table_header(*header_cols)
            for _, row in ci_df.iterrows():
                mc_i = int(row.get("candidate_mc_idx", 0))
                cells = [str(mc_i)]
                if has_tiers:
                    cells.append(_tier_lookup.get(mc_i, "?"))
                for c in ci_cols[:12]:
                    cells.append(_fmt_num(row.get(c), 3))
                table_row(*cells)
            p("")

        bootstrap = uncertainty_results.get("bootstrap")
        if bootstrap is not None:
            stable = bootstrap.get("stable_front", [])
            n_res = bootstrap.get("n_resample", 0)
            h3("Pareto Front Stability (Bootstrap)")
            p(f"Bootstrap resamples: {n_res}")
            p(f"Stable front members (>=50% membership): {len(stable)} candidates")
            if stable:
                p(f"Stable MC indices: {stable[:20]}{'...' if len(stable) > 20 else ''}")
            p("")

    # ── Robustness analysis ──
    if robustness_results is not None:
        h2("Robustness Analysis")

        convergence_df = robustness_results.get("convergence")
        if convergence_df is not None and not convergence_df.empty:
            h3("N-Convergence")
            metric_cols = [c for c in convergence_df.columns
                           if c not in ("N", "n_ok") and not c.startswith("converged_")]
            header_cols = ["N", "n_ok"] + metric_cols
            table_header(*header_cols)
            for _, row in convergence_df.iterrows():
                cells = [str(int(row["N"])), str(int(row["n_ok"]))]
                for c in metric_cols:
                    cells.append(_fmt_num(row.get(c), 4))
                table_row(*cells)
            p("")

            # Convergence verdict
            for m in metric_cols:
                conv_col = f"converged_{m}"
                if conv_col in convergence_df.columns:
                    converged_at = convergence_df.loc[convergence_df[conv_col] == True, "N"]
                    if not converged_at.empty:
                        p(f"- `{m}` converged (<1% change) at N={int(converged_at.iloc[0])}")
                    else:
                        p(f"- `{m}` did NOT converge within tested range")
            p("")

        sensitivity_df = robustness_results.get("sensitivity")
        if sensitivity_df is not None and not sensitivity_df.empty:
            h3("Numerical Sensitivity")
            metric_cols = [c for c in sensitivity_df.columns
                           if c not in ("parameter", "value", "n_ok")
                           and not c.startswith("pct_change_")]
            pct_cols = [c for c in sensitivity_df.columns if c.startswith("pct_change_")]
            header_cols = ["parameter", "value", "n_ok"] + metric_cols + pct_cols
            table_header(*header_cols)
            for _, row in sensitivity_df.iterrows():
                cells = [str(row["parameter"]), str(row["value"]), str(int(row["n_ok"]))]
                for c in metric_cols:
                    cells.append(_fmt_num(row.get(c), 4))
                for c in pct_cols:
                    v = row.get(c)
                    cells.append(f"{v:+.2f}%" if isinstance(v, float) and np.isfinite(v) else "n/a")
                table_row(*cells)
            p("")

    # Closing summary
    h2("Conclusions")
    if best_ana is None and best_vec_ana is None:
        p("No valid candidate could be promoted to the final diagnostics stage.")
    else:
        if best_ana is not None:
            tier_lbl = best_ana.get("tier", "unknown")
            p(
                f"- Best scalar-speed outcome: MC#{best_idx}, Delta-v={_fmt_num(best_ana.get('delta_v'), 4)} km/s "
                f"({_fmt_num(best_ana.get('delta_v_pct'), 2)}%), tier={tier_lbl}."
            )
        if best_vec_ana is not None:
            tier_lbl = best_vec_ana.get("tier", "unknown")
            p(
                f"- Best vector-turn outcome: MC#{best_vec_idx}, DeltaV_vec={_fmt_num(best_vec_ana.get('delta_v_vec'), 4)} km/s, "
                f"0.5*DeltaV^2={_fmt_num(best_vec_ana.get('energy_half_dv_vec_sq'), 4)} km^2/s^2, tier={tier_lbl}."
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
    from .. import __version__
    p(f"*slingshot-solver v{__version__}*")

    close_table()
    report_text = "\n".join(lines)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text
