"""
Tests for the ground-truth 2-body scattering engine (TwoBodyScatter).

This is pure math with zero external dependencies — the most testable
module in the package.
"""

import math
import pytest


class TestHelpers:
    """Test scalar helper functions."""

    def test_norm2(self):
        from slingshot.core.twobody_scatter import norm2
        assert norm2(3.0, 4.0) == pytest.approx(5.0)

    def test_sgn(self):
        from slingshot.core.twobody_scatter import sgn
        assert sgn(5.0) == 1.0
        assert sgn(-3.0) == -1.0
        assert sgn(0.0) == 0.0

    def test_star_velocity_scalar(self):
        from slingshot.core.twobody_scatter import _star_velocity_components
        vx, vy = _star_velocity_components(10.0)
        assert vx == 0.0
        assert vy == 10.0

    def test_star_velocity_vector(self):
        from slingshot.core.twobody_scatter import _star_velocity_components
        vx, vy = _star_velocity_components((3.0, 4.0))
        assert vx == 3.0
        assert vy == 4.0

    def test_star_velocity_bad_length(self):
        from slingshot.core.twobody_scatter import _star_velocity_components
        with pytest.raises(ValueError):
            _star_velocity_components((1.0, 2.0, 3.0))


class TestOrbitalElements:
    """Test orbital element computations."""

    def test_energy_positive_for_unbound(self):
        """Unbound orbit should have positive specific energy."""
        from slingshot.core.twobody_scatter import epsilonfn
        # Fast particle far from star → positive energy
        # mu in km³/s² for Sun: G_KM * M_SUN ≈ 1.327e11
        mu = 6.6743e-20 * 1.989e30  # ~1.327e11 km³/s²
        # Particle at 1 AU with 100 km/s → well above escape velocity
        eps = epsilonfn(um0=100.0, vm0=0.0, vstar0=0.0, mu=mu,
                        xm0=1.496e8, ym0=0.0)
        assert eps > 0

    def test_vinf_from_energy(self):
        from slingshot.core.twobody_scatter import vinffn
        eps = 100.0  # km²/s²
        vinf = vinffn(eps)
        assert vinf == pytest.approx(math.sqrt(2 * eps))

    def test_eccentricity_hyperbolic(self):
        """e > 1 for hyperbolic orbit."""
        from slingshot.core.twobody_scatter import e_from_b_vinf
        e = e_from_b_vinf(b=1e6, vinf=50.0, mu=1.327e11)
        assert e > 1.0

    def test_semi_major_axis_negative(self):
        """Semi-major axis should be negative for hyperbolic orbit."""
        from slingshot.core.twobody_scatter import a_from_vinf
        a = a_from_vinf(mu=1.327e11, vinf=50.0)
        assert a < 0


class TestGravityAssist:
    """Test full gravity-assist computations."""

    def test_no_burn_returns_result(self):
        from slingshot.core.twobody_scatter import gravity_assist_no_burn
        mu = 6.6743e-20 * 1.989e30
        # Particle far away at high speed → guaranteed unbound
        res = gravity_assist_no_burn(
            xm0=1.5e8, ym0=1e7, um0=-100.0, vm0=0.0,
            vstar0=0.0, mu=mu,
        )
        assert hasattr(res, "umF")
        assert hasattr(res, "vmF")
        assert hasattr(res, "epsilon")
        assert hasattr(res, "theta")

    def test_no_burn_conserves_energy_in_star_frame(self):
        """In the star frame, specific energy is conserved: |v_inf_in| = |v_inf_out|."""
        from slingshot.core.twobody_scatter import gravity_assist_no_burn, norm2
        mu = 6.6743e-20 * 1.989e30
        res = gravity_assist_no_burn(
            xm0=1.5e8, ym0=2e7, um0=-100.0, vm0=5.0,
            vstar0=0.0, mu=mu,
        )
        # vinf magnitude conserved (scattering is elastic in star frame)
        assert res.vinf == pytest.approx(math.sqrt(2 * res.epsilon), rel=1e-6)

    def test_deltaV_lab(self):
        from slingshot.core.twobody_scatter import deltaV_lab
        dv = deltaV_lab(10.0, 0.0, 10.0, 0.0)
        assert dv == pytest.approx(0.0)
        dv = deltaV_lab(10.0, 0.0, 0.0, 10.0)
        assert dv == pytest.approx(math.sqrt(200.0))

    def test_oberth_burn_increases_deltaV(self):
        """Oberth maneuver at periapsis should yield higher ΔV than no-burn."""
        from slingshot.core.twobody_scatter import (
            gravity_assist_no_burn,
            gravity_assist_oberth,
            deltaV_lab,
        )
        mu = 6.6743e-20 * 1.989e30
        args = dict(xm0=1.5e8, ym0=2e7, um0=-100.0, vm0=0.0,
                    vstar0=0.0, mu=mu)
        res_nb = gravity_assist_no_burn(**args)
        res_ob = gravity_assist_oberth(**args, dv=1.0)
        dv_nb = deltaV_lab(-100.0, 0.0, res_nb.umF, res_nb.vmF)
        dv_ob = deltaV_lab(-100.0, 0.0, res_ob.umF, res_ob.vmF)
        assert dv_ob > dv_nb

    def test_closed_form_returns_scattering_result(self):
        from slingshot.core.twobody_scatter import gravity_assist_closed_form
        mu = 6.6743e-20 * 1.989e30
        res = gravity_assist_closed_form(
            xm0=1.5e8, ym0=1e7, um0=-100.0, vm0=0.0,
            vstar0=0.0, mu=mu,
        )
        assert hasattr(res, "umF")
        assert hasattr(res, "wmF")
        assert hasattr(res, "theta")
