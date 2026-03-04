"""
Tests for uncertainty propagation and Pareto stability analysis.
"""

import pytest
import numpy as np
import pandas as pd


class TestBootstrapParetoStability:
    """Test bootstrap_pareto_stability with synthetic MC data."""

    def _make_synthetic_mc(self, n=100, seed=42):
        """Create a minimal synthetic MC dict."""
        rng = np.random.default_rng(seed)
        ok = np.ones(n, dtype=bool)
        ok[:10] = False  # 10 failures
        return {
            "ok": ok,
            "delta_v": rng.normal(5.0, 2.0, n),
            "delta_v_pct": rng.normal(10.0, 5.0, n),
            "delta_v_vec": rng.normal(8.0, 3.0, n),
            "energy_half_dv_vec_sq": rng.normal(32.0, 10.0, n),
            "deflection": rng.normal(30.0, 15.0, n),
            "r_min": rng.uniform(1e5, 1e7, n),
            "r_star_min": rng.uniform(1e6, 1e8, n),
            "delta_v_planet_frame": rng.normal(3.0, 1.0, n),
            "energy_from_planet_orbit": rng.normal(10.0, 5.0, n),
        }

    def test_returns_expected_keys(self):
        from slingshot.analysis.uncertainty import bootstrap_pareto_stability

        mc = self._make_synthetic_mc()
        objectives = [
            {"metric": "delta_v", "sign": "maximize", "weight": 1.0},
            {"metric": "delta_v_vec", "sign": "maximize", "weight": 1.0},
        ]

        result = bootstrap_pareto_stability(
            mc, objectives=objectives, n_resample=20, seed=123,
        )

        assert "membership_freq" in result
        assert "stable_front" in result
        assert "n_resample" in result
        assert result["n_resample"] == 20

    def test_frequencies_are_valid(self):
        from slingshot.analysis.uncertainty import bootstrap_pareto_stability

        mc = self._make_synthetic_mc()
        objectives = [
            {"metric": "delta_v", "sign": "maximize", "weight": 1.0},
            {"metric": "delta_v_vec", "sign": "maximize", "weight": 1.0},
        ]

        result = bootstrap_pareto_stability(
            mc, objectives=objectives, n_resample=50, seed=42,
        )

        freq = result["membership_freq"]
        for idx, f in freq.items():
            assert 0.0 <= f <= 1.0, f"Frequency {f} out of range for index {idx}"

    def test_stable_front_subset_of_membership(self):
        from slingshot.analysis.uncertainty import bootstrap_pareto_stability

        mc = self._make_synthetic_mc()
        objectives = [
            {"metric": "delta_v", "sign": "maximize", "weight": 1.0},
        ]

        result = bootstrap_pareto_stability(
            mc, objectives=objectives, n_resample=30, seed=7,
        )

        stable = set(result["stable_front"])
        freq_keys = set(result["membership_freq"].keys())
        assert stable.issubset(freq_keys)

    def test_empty_mc(self):
        from slingshot.analysis.uncertainty import bootstrap_pareto_stability

        mc = {"ok": np.zeros(10, dtype=bool), "delta_v": np.zeros(10)}
        objectives = [{"metric": "delta_v", "sign": "maximize", "weight": 1.0}]

        result = bootstrap_pareto_stability(mc, objectives=objectives, n_resample=5, seed=1)
        assert result["membership_freq"] == {}
        assert result["stable_front"] == []


class TestConfidenceBands:
    """Test compute_confidence_bands with known distributions."""

    def test_basic_ci(self):
        from slingshot.analysis.uncertainty import compute_confidence_bands

        rng = np.random.default_rng(42)
        n_draws = 200
        df = pd.DataFrame({
            "draw": np.repeat(range(n_draws), 2),
            "candidate_mc_idx": np.tile([0, 1], n_draws),
            "delta_v": rng.normal(10.0, 1.0, n_draws * 2),
            "delta_v_vec": rng.normal(15.0, 2.0, n_draws * 2),
        })

        ci = compute_confidence_bands(df, ci_levels=[0.68, 0.95])

        assert len(ci) == 2  # two candidates
        assert "delta_v_median" in ci.columns
        assert "delta_v_lo68" in ci.columns
        assert "delta_v_hi68" in ci.columns
        assert "delta_v_lo95" in ci.columns
        assert "delta_v_hi95" in ci.columns

        # 95% CI should be wider than 68% CI
        for _, row in ci.iterrows():
            assert row["delta_v_lo95"] <= row["delta_v_lo68"]
            assert row["delta_v_hi95"] >= row["delta_v_hi68"]

    def test_single_candidate(self):
        from slingshot.analysis.uncertainty import compute_confidence_bands

        df = pd.DataFrame({
            "draw": range(50),
            "candidate_mc_idx": [0] * 50,
            "delta_v": np.linspace(5.0, 15.0, 50),
        })

        ci = compute_confidence_bands(df, metric_names=["delta_v"])
        assert len(ci) == 1
        assert ci.iloc[0]["delta_v_median"] == pytest.approx(10.0, abs=0.5)

    def test_empty_df(self):
        from slingshot.analysis.uncertainty import compute_confidence_bands

        df = pd.DataFrame(columns=["draw", "candidate_mc_idx", "delta_v"])
        ci = compute_confidence_bands(df)
        assert len(ci) == 0


class TestDrawSystemParams:
    """Test _draw_system_params parameter sampling."""

    def test_draws_vary(self):
        from slingshot.analysis.uncertainty import _draw_system_params
        from slingshot.config import FullConfig

        cfg = FullConfig()
        rng = np.random.default_rng(42)
        dists = {"M_planet_Mjup": {"mean": 5.2, "std": 0.4}}

        masses = []
        for _ in range(20):
            drawn = _draw_system_params(cfg, rng, dists)
            masses.append(drawn.system.M_planet_Mjup)

        # Should have variation
        assert np.std(masses) > 0.01

    def test_unknown_field_raises(self):
        from slingshot.analysis.uncertainty import _draw_system_params
        from slingshot.config import FullConfig

        cfg = FullConfig()
        rng = np.random.default_rng(1)
        dists = {"nonexistent_field": {"mean": 1.0, "std": 0.1}}

        with pytest.raises(KeyError, match="nonexistent_field"):
            _draw_system_params(cfg, rng, dists)
