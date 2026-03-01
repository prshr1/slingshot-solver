"""
Tests for 3-body dynamics integration.
"""

import numpy as np
import pytest


class TestInitHotJupiter:
    """Test barycentric initial condition generation."""

    def test_returns_8_component_state(self):
        from slingshot.core.dynamics import init_hot_jupiter_barycentric
        Y = init_hot_jupiter_barycentric()
        assert Y.shape == (8,)

    def test_barycenter_at_origin(self):
        from slingshot.core.dynamics import init_hot_jupiter_barycentric
        from slingshot.constants import M_SUN, M_JUP
        Y = init_hot_jupiter_barycentric()
        xs, ys = Y[0], Y[1]
        xp, yp = Y[4], Y[5]
        # COM should be at origin
        M_tot = M_SUN + M_JUP
        x_com = (M_SUN * xs + M_JUP * xp) / M_tot
        y_com = (M_SUN * ys + M_JUP * yp) / M_tot
        assert abs(x_com) < 1e-6  # km
        assert abs(y_com) < 1e-6

    def test_bulk_velocity_added(self):
        from slingshot.core.dynamics import init_hot_jupiter_barycentric
        Y0 = init_hot_jupiter_barycentric(bulk_velocity_vx_kms=0.0,
                                          bulk_velocity_vy_kms=0.0)
        Y1 = init_hot_jupiter_barycentric(bulk_velocity_vx_kms=10.0,
                                          bulk_velocity_vy_kms=5.0)
        # Both star and planet velocities should shift by (10, 5)
        assert Y1[2] - Y0[2] == pytest.approx(10.0)
        assert Y1[3] - Y0[3] == pytest.approx(5.0)
        assert Y1[6] - Y0[6] == pytest.approx(10.0)
        assert Y1[7] - Y0[7] == pytest.approx(5.0)


class TestSimulate3Body:
    """Smoke tests for the ODE integrator."""

    def test_integration_succeeds(self, sample_star_planet_state):
        from slingshot.core.dynamics import simulate_3body
        Y_sp = sample_star_planet_state
        # Add a satellite approaching from far away
        Y0 = np.concatenate([Y_sp, [5e7, 1e7, -30.0, 0.0]])
        sol = simulate_3body(Y0, (0, 1e6), n_eval=100)
        assert sol is not None
        assert sol.y.shape[0] == 12
        assert sol.y.shape[1] >= 2

    def test_escape_radius_terminates_early(self, sample_star_planet_state):
        from slingshot.core.dynamics import simulate_3body
        Y_sp = sample_star_planet_state
        Y0 = np.concatenate([Y_sp, [5e7, 0.0, -50.0, 0.0]])
        sol = simulate_3body(Y0, (0, 1e8), escape_radius_km=1e8)
        assert sol is not None
        # Should terminate before the full time span
        assert sol.t[-1] < 1e8
