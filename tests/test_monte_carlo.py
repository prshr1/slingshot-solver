"""
Tests for Monte Carlo single-particle evaluation.
"""

import numpy as np
import pytest


class TestEvaluateParticle:
    """Smoke test for single-particle MC evaluation."""

    def test_single_particle(self, sample_star_planet_state):
        from slingshot.analysis.monte_carlo import evaluate_particle
        from slingshot.constants import M_SUN, M_JUP, R_JUP

        Y_sp = sample_star_planet_state
        sat = np.array([5e7, 1e7, -30.0, 0.0])

        # evaluate_particle expects a single tuple argument
        particle_args = (
            0,             # idx
            sat,           # sat_state
            Y_sp,          # Y_sp0
            (0, 2e6),      # t_span
            M_SUN,         # m_star
            M_JUP,         # m_p
            R_JUP,         # R_p
            "barycentric", # frame
            {},            # ana_kwargs
        )
        result = evaluate_particle(particle_args)
        assert isinstance(result, dict)
        assert "idx" in result
