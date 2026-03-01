"""
Tests for trajectory analysis.
"""

import numpy as np
import pytest


class TestAnalyzeTrajectory:
    """Smoke tests for trajectory analysis on synthetic data."""

    def test_analyze_returns_dict(self, sample_star_planet_state):
        from slingshot.core.dynamics import simulate_3body
        from slingshot.analysis.trajectory import analyze_trajectory

        Y_sp = sample_star_planet_state
        Y0 = np.concatenate([Y_sp, [5e7, 1e7, -30.0, 0.0]])
        sol = simulate_3body(Y0, (0, 2e6), n_eval=500)
        assert sol is not None

        result = analyze_trajectory(sol)
        assert isinstance(result, dict)
        assert "delta_v" in result or "encounter" in result


class TestEncounterGeometry:
    """Test EncounterGeometry dataclass."""

    def test_failed_geometry(self):
        from slingshot.analysis.trajectory import EncounterGeometry
        eg = EncounterGeometry(ok=False, reason="test failure")
        assert not eg.ok
        assert eg.reason == "test failure"

    def test_successful_geometry(self):
        from slingshot.analysis.trajectory import EncounterGeometry
        eg = EncounterGeometry(ok=True, i0=0, i1=100, k_min=50)
        assert eg.ok
        assert eg.k_min == 50
