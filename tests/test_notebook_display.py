"""Tests for the notebook_display module and compare_runs enhancements."""

import numpy as np
import pytest

# ───────────────────────────────────────────────────────────────────
# Skip entire module if pandas not available
# ───────────────────────────────────────────────────────────────────

pd = pytest.importorskip("pandas")

from slingshot.output.notebook_display import (
    candidates_dataframe,
    styled_candidates_table,
    styled_cross_run_table,
    styled_tier_summary,
    styled_uncertainty_table,
)
from slingshot.output.compare_runs import (
    build_comparison_df,
    discover_runs,
)
from slingshot.analysis.tiering import (
    TIER_HYBRID,
    TIER_ORDER,
    TIER_PLANET,
    TIER_STAR,
    annotate_tiers,
    tier_candidates,
    tier_summary_stats,
)


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

def _make_analysis(eps_planet, d_eps_mono, delta_v=10.0, **extra):
    ana = {
        "energy_from_planet_orbit": eps_planet,
        "delta_eps_monopole": d_eps_mono,
        "delta_v": delta_v,
        "delta_v_pct": delta_v * 0.1,
        "delta_v_vec": delta_v * 0.9,
        "energy_half_dv_vec_sq": 0.5 * (delta_v * 0.9) ** 2,
        "deflection": 15.0,
        "r_min": 1e5,
        "encounter": None,
    }
    ana.update(extra)
    return ana


def _annotated_analyses():
    """Build a small list of analyses with tiers annotated."""
    analyses = [
        _make_analysis(100, 10, delta_v=50),    # planet
        _make_analysis(10, 10, delta_v=40),      # hybrid
        _make_analysis(1, 100, delta_v=30),      # star
        None,
        _make_analysis(200, 10, delta_v=20),     # planet
    ]
    annotate_tiers(analyses)
    return analyses


# ───────────────────────────────────────────────────────────────────
# candidates_dataframe
# ───────────────────────────────────────────────────────────────────

class TestCandidatesDataFrame:
    def test_basic_dataframe(self):
        analyses = _annotated_analyses()
        df = candidates_dataframe(analyses, [10, 20, 30, 40, 50])
        assert len(df) == 4  # 4 valid (1 None skipped)
        assert "Δv (km/s)" in df.columns
        assert "Tier" in df.columns
        # Sorted by Δv descending
        assert df.iloc[0]["Δv (km/s)"] == 50.0

    def test_top_n_limits(self):
        analyses = _annotated_analyses()
        df = candidates_dataframe(analyses, top_n=2)
        assert len(df) == 2

    def test_empty_input(self):
        df = candidates_dataframe([None, None])
        assert len(df) == 0

    def test_no_top_indices(self):
        analyses = _annotated_analyses()
        df = candidates_dataframe(analyses)
        assert len(df) == 4
        # MC# should be local index
        assert df.iloc[0]["MC#"] == 0  # local index of the highest Δv


# ───────────────────────────────────────────────────────────────────
# styled_candidates_table
# ───────────────────────────────────────────────────────────────────

class TestStyledCandidatesTable:
    def test_returns_styler(self):
        analyses = _annotated_analyses()
        styler = styled_candidates_table(analyses, [10, 20, 30, 40, 50])
        assert hasattr(styler, "to_html")

    def test_empty_returns_styler(self):
        styler = styled_candidates_table([None])
        assert hasattr(styler, "to_html")


# ───────────────────────────────────────────────────────────────────
# styled_tier_summary
# ───────────────────────────────────────────────────────────────────

class TestStyledTierSummary:
    def test_tier_summary_table(self):
        analyses = _annotated_analyses()
        tiered = tier_candidates(analyses, [10, 20, 30, 40, 50])
        stats = tier_summary_stats(tiered)
        styler = styled_tier_summary(stats)
        assert hasattr(styler, "to_html")
        # Check underlying data
        df = styler.data
        assert len(df) == 3  # one row per tier
        assert df.iloc[0]["Tier"] == TIER_PLANET


# ───────────────────────────────────────────────────────────────────
# styled_cross_run_table
# ───────────────────────────────────────────────────────────────────

class TestStyledCrossRunTable:
    def test_empty_df(self):
        df = pd.DataFrame({"run": [], "best_dv_kms": []})
        styler = styled_cross_run_table(df)
        assert hasattr(styler, "to_html")

    def test_with_data(self):
        df = pd.DataFrame([
            {"run": "run_a", "best_dv_kms": 10.0, "n_candidates": 5},
            {"run": "run_b", "best_dv_kms": 20.0, "n_candidates": 10},
        ])
        styler = styled_cross_run_table(df)
        html = styler.to_html()
        assert "run_a" in html
        assert "run_b" in html


# ───────────────────────────────────────────────────────────────────
# styled_uncertainty_table
# ───────────────────────────────────────────────────────────────────

class TestStyledUncertaintyTable:
    def test_basic(self):
        ci_df = pd.DataFrame([
            {"candidate_mc_idx": 100, "dv_median": 10.0, "dv_lo68": 9.5, "dv_hi68": 10.5},
            {"candidate_mc_idx": 200, "dv_median": 8.0, "dv_lo68": 7.5, "dv_hi68": 8.5},
        ])
        styler = styled_uncertainty_table(ci_df)
        assert hasattr(styler, "to_html")

    def test_with_tier_lookup(self):
        ci_df = pd.DataFrame([
            {"candidate_mc_idx": 100, "dv_median": 10.0},
            {"candidate_mc_idx": 200, "dv_median": 8.0},
        ])
        tier_lk = {100: TIER_PLANET, 200: TIER_STAR}
        styler = styled_uncertainty_table(ci_df, tier_lookup=tier_lk)
        html = styler.to_html()
        assert "Tier" in html

    def test_empty(self):
        ci_df = pd.DataFrame()
        styler = styled_uncertainty_table(ci_df)
        assert hasattr(styler, "to_html")


# ───────────────────────────────────────────────────────────────────
# compare_runs enhancements
# ───────────────────────────────────────────────────────────────────

class TestDiscoverRuns:
    def test_empty_dir(self, tmp_path):
        assert discover_runs(str(tmp_path)) == []

    def test_finds_results_dirs(self, tmp_path):
        (tmp_path / "results_A").mkdir()
        (tmp_path / "results_B").mkdir()
        (tmp_path / "other_dir").mkdir()
        found = discover_runs(str(tmp_path))
        assert len(found) == 2

    def test_system_filter(self, tmp_path):
        (tmp_path / "results_Kepler-432_001").mkdir()
        (tmp_path / "results_TOI-1431_001").mkdir()
        found = discover_runs(str(tmp_path), system_filter="Kepler")
        assert len(found) == 1
        assert "Kepler" in found[0]

    def test_nonexistent_root(self):
        assert discover_runs("/nonexistent/path_xyz") == []


class TestBuildComparisonDf:
    def test_empty_dirs(self):
        df = build_comparison_df([])
        assert len(df) == 0

    def test_nonexistent_dirs(self):
        df = build_comparison_df(["/nonexistent_dir_xyz"])
        assert len(df) == 0
