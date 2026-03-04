"""Tests for the physics-based candidate tiering module."""

import numpy as np
import pytest

from slingshot.analysis.tiering import (
    TIER_HYBRID,
    TIER_ORDER,
    TIER_PLANET,
    TIER_STAR,
    annotate_tiers,
    classify_candidate,
    tier_candidates,
    tier_ratio,
    tier_summary_stats,
)


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

def _make_analysis(eps_planet, d_eps_mono, delta_v=10.0, **extra):
    """Helper to build a minimal analysis dict."""
    ana = {
        "energy_from_planet_orbit": eps_planet,
        "delta_eps_monopole": d_eps_mono,
        "delta_v": delta_v,
        "delta_v_pct": delta_v * 0.1,
        "deflection": 15.0,
        "r_min": 1e5,
        "delta_v_vec": delta_v * 0.9,
        "energy_half_dv_vec_sq": 0.5 * (delta_v * 0.9) ** 2,
    }
    ana.update(extra)
    return ana


# ───────────────────────────────────────────────────────────────────
# classify_candidate tests
# ───────────────────────────────────────────────────────────────────

class TestClassifyCandidate:
    def test_planet_dominated_high_ratio(self):
        ana = _make_analysis(eps_planet=100.0, d_eps_mono=10.0)
        # ratio = 10.0 > 2.0
        assert classify_candidate(ana) == TIER_PLANET

    def test_hybrid_medium_ratio(self):
        ana = _make_analysis(eps_planet=10.0, d_eps_mono=10.0)
        # ratio = 1.0, between 0.5 and 2.0
        assert classify_candidate(ana) == TIER_HYBRID

    def test_star_dominated_low_ratio(self):
        ana = _make_analysis(eps_planet=2.0, d_eps_mono=100.0)
        # ratio = 0.02 < 0.5
        assert classify_candidate(ana) == TIER_STAR

    def test_exact_threshold_planet(self):
        # ratio = 2.0 → NOT planet (needs > 2.0), should be hybrid
        ana = _make_analysis(eps_planet=20.0, d_eps_mono=10.0)
        assert classify_candidate(ana) == TIER_HYBRID

    def test_exact_threshold_hybrid(self):
        # ratio = 0.5 → hybrid (≥ 0.5)
        ana = _make_analysis(eps_planet=5.0, d_eps_mono=10.0)
        assert classify_candidate(ana) == TIER_HYBRID

    def test_negative_eps_planet(self):
        ana = _make_analysis(eps_planet=-5.0, d_eps_mono=10.0)
        assert classify_candidate(ana) == TIER_STAR

    def test_zero_monopole(self):
        # Δε_monopole ≈ 0, ε_planet > 0 → planet
        ana = _make_analysis(eps_planet=5.0, d_eps_mono=0.0)
        assert classify_candidate(ana) == TIER_PLANET

    def test_both_zero(self):
        ana = _make_analysis(eps_planet=0.0, d_eps_mono=0.0)
        # ε_planet = 0 → star-dominated (negative branch hits first)
        assert classify_candidate(ana) == TIER_STAR

    def test_missing_fields(self):
        assert classify_candidate({}) == TIER_STAR
        assert classify_candidate({"energy_from_planet_orbit": 5.0}) == TIER_STAR

    def test_nan_values(self):
        ana = _make_analysis(eps_planet=float("nan"), d_eps_mono=10.0)
        assert classify_candidate(ana) == TIER_STAR

    def test_inf_eps_planet(self):
        ana = _make_analysis(eps_planet=float("inf"), d_eps_mono=10.0)
        assert classify_candidate(ana) == TIER_STAR

    def test_custom_thresholds(self):
        ana = _make_analysis(eps_planet=15.0, d_eps_mono=10.0)
        # ratio = 1.5; default is hybrid, but with threshold=1.0 → planet
        assert classify_candidate(ana, planet_threshold=1.0) == TIER_PLANET
        # With very high threshold → star-dominated
        assert classify_candidate(ana, planet_threshold=5.0, hybrid_threshold=3.0) == TIER_STAR

    def test_negative_monopole(self):
        # abs(d_eps_mono) = 50; ratio = 100/50 = 2.0 → hybrid (not > 2.0)
        ana = _make_analysis(eps_planet=100.0, d_eps_mono=-50.0)
        assert classify_candidate(ana) == TIER_HYBRID

        # ratio = 100/20 = 5.0 → planet
        ana2 = _make_analysis(eps_planet=100.0, d_eps_mono=-20.0)
        assert classify_candidate(ana2) == TIER_PLANET


# ───────────────────────────────────────────────────────────────────
# tier_ratio tests
# ───────────────────────────────────────────────────────────────────

class TestTierRatio:
    def test_normal_ratio(self):
        ana = _make_analysis(eps_planet=30.0, d_eps_mono=10.0)
        assert tier_ratio(ana) == pytest.approx(3.0)

    def test_zero_monopole_positive_planet(self):
        ana = _make_analysis(eps_planet=5.0, d_eps_mono=0.0)
        assert tier_ratio(ana) == float("inf")

    def test_zero_monopole_zero_planet(self):
        ana = _make_analysis(eps_planet=0.0, d_eps_mono=0.0)
        assert tier_ratio(ana) == 0.0

    def test_missing_returns_none(self):
        assert tier_ratio({}) is None

    def test_nan_returns_none(self):
        ana = _make_analysis(eps_planet=float("nan"), d_eps_mono=10.0)
        assert tier_ratio(ana) is None


# ───────────────────────────────────────────────────────────────────
# tier_candidates tests
# ───────────────────────────────────────────────────────────────────

class TestTierCandidates:
    def test_buckets_all_tiers(self):
        analyses = [
            _make_analysis(100, 10, delta_v=50),     # ratio=10 → planet
            _make_analysis(10, 10, delta_v=40),       # ratio=1 → hybrid
            _make_analysis(1, 100, delta_v=30),       # ratio=0.01 → star
            None,                                      # skipped
            _make_analysis(200, 10, delta_v=20),      # ratio=20 → planet
        ]
        top_indices = [100, 200, 300, 400, 500]
        result = tier_candidates(analyses, top_indices)

        assert len(result[TIER_PLANET]) == 2
        assert len(result[TIER_HYBRID]) == 1
        assert len(result[TIER_STAR]) == 1

        # Planet bucket sorted by Δv descending
        assert result[TIER_PLANET][0][1] == 100  # MC#100, Δv=50
        assert result[TIER_PLANET][1][1] == 500  # MC#500, Δv=20

    def test_ranks_are_sequential(self):
        analyses = [
            _make_analysis(100, 10, delta_v=50),
            _make_analysis(200, 10, delta_v=40),
            _make_analysis(300, 10, delta_v=30),
        ]
        result = tier_candidates(analyses)
        ranks = [r for r, _, _ in result[TIER_PLANET]]
        assert ranks == [1, 2, 3]

    def test_empty_input(self):
        result = tier_candidates([])
        for tier in TIER_ORDER:
            assert result[tier] == []

    def test_all_none(self):
        result = tier_candidates([None, None, None])
        for tier in TIER_ORDER:
            assert result[tier] == []


# ───────────────────────────────────────────────────────────────────
# tier_summary_stats tests
# ───────────────────────────────────────────────────────────────────

class TestTierSummaryStats:
    def test_stats_computed(self):
        analyses = [
            _make_analysis(100, 10, delta_v=50),
            _make_analysis(200, 10, delta_v=30),
            _make_analysis(10, 10, delta_v=40),
        ]
        tiered = tier_candidates(analyses, [10, 20, 30])
        stats = tier_summary_stats(tiered)

        assert stats[TIER_PLANET]["count"] == 2
        assert stats[TIER_PLANET]["dv_max"] == 50.0
        assert stats[TIER_PLANET]["dv_min"] == 30.0
        assert stats[TIER_HYBRID]["count"] == 1
        assert stats[TIER_STAR]["count"] == 0
        assert stats[TIER_STAR]["dv_max"] is None

    def test_empty_tiers(self):
        stats = tier_summary_stats({t: [] for t in TIER_ORDER})
        for tier in TIER_ORDER:
            assert stats[tier]["count"] == 0
            assert stats[tier]["top_mc_idx"] is None


# ───────────────────────────────────────────────────────────────────
# annotate_tiers tests
# ───────────────────────────────────────────────────────────────────

class TestAnnotateTiers:
    def test_adds_tier_and_ratio(self):
        analyses = [
            _make_analysis(100, 10, delta_v=50),
            None,
            _make_analysis(1, 100, delta_v=20),
        ]
        annotate_tiers(analyses)

        assert analyses[0]["tier"] == TIER_PLANET
        assert analyses[0]["tier_ratio"] == pytest.approx(10.0)
        assert analyses[1] is None
        assert analyses[2]["tier"] == TIER_STAR
        assert analyses[2]["tier_ratio"] == pytest.approx(0.01)

    def test_custom_thresholds(self):
        analyses = [_make_analysis(15, 10)]  # ratio=1.5
        annotate_tiers(analyses, planet_threshold=1.0)
        assert analyses[0]["tier"] == TIER_PLANET

    def test_idempotent(self):
        analyses = [_make_analysis(100, 10)]
        annotate_tiers(analyses)
        tier1 = analyses[0]["tier"]
        annotate_tiers(analyses)
        assert analyses[0]["tier"] == tier1
