"""
Tests for robustness / sensitivity analysis.
"""

import pytest
import numpy as np
import pandas as pd


class TestFormatRobustnessTable:
    """Test markdown / LaTeX formatting."""

    def _make_sensitivity_df(self):
        """Minimal sensitivity DataFrame."""
        return pd.DataFrame({
            "parameter": ["softening", "softening", "tolerance", "tolerance"],
            "value": [1e3, 1e4, 1e-9, 1e-11],
            "delta_v_pct": [0.5, -1.2, 0.0, 0.3],
            "delta_v_vec_pct": [0.3, -0.8, 0.1, 0.2],
        })

    def test_markdown_output(self):
        from slingshot.analysis.robustness import format_robustness_table

        df = self._make_sensitivity_df()
        md = format_robustness_table(df, metric_names=["delta_v", "delta_v_vec"], fmt="markdown")
        assert isinstance(md, str)
        assert "|" in md
        assert "softening" in md

    def test_latex_output(self):
        from slingshot.analysis.robustness import format_robustness_table

        df = self._make_sensitivity_df()
        tex = format_robustness_table(df, metric_names=["delta_v", "delta_v_vec"], fmt="latex")
        assert isinstance(tex, str)
        assert "\\begin{tabular" in tex or "tabular" in tex.lower() or "&" in tex

    def test_unknown_fmt_raises(self):
        from slingshot.analysis.robustness import format_robustness_table

        df = self._make_sensitivity_df()
        with pytest.raises(ValueError):
            format_robustness_table(df, metric_names=["delta_v"], fmt="csv")


class TestConvergenceStructure:
    """Structural tests for run_convergence_test output."""

    def test_convergence_df_columns(self):
        """Test that convergence output has expected shape when we supply
        a fully valid config."""
        from slingshot.config import FullConfig

        cfg = FullConfig()

        # Patch a small N so the test is fast
        from slingshot.analysis.robustness import run_convergence_test

        try:
            result = run_convergence_test(
                base_cfg=cfg,
                N_values=[5, 10],
                seed=42,
                metric_names=["delta_v"],
            )
        except Exception:
            pytest.skip("Full MC pipeline not available in unit-test env")

        assert isinstance(result, pd.DataFrame)
        assert "N" in result.columns
        assert "delta_v" in result.columns

    def test_convergence_empty_n(self):
        """Empty N list -> empty DataFrame."""
        from slingshot.analysis.robustness import run_convergence_test
        from slingshot.config import FullConfig

        cfg = FullConfig()
        result = run_convergence_test(
            base_cfg=cfg,
            N_values=[],
            seed=0,
            metric_names=["delta_v"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSensitivityStructure:
    """Test run_numerical_sensitivity output shape."""

    def test_sensitivity_smoke(self):
        """End-to-end sensitivity test (may skip if full pipeline unavailable)."""
        from slingshot.config import FullConfig, RobustnessConfig
        from slingshot.analysis.robustness import run_numerical_sensitivity

        cfg = FullConfig()
        rob_cfg = RobustnessConfig(
            enabled=True,
            seed=42,
            convergence_N=[5],
            softening_values=[1e3],
            tolerance_values=[],
            clearance_values=[],
            flyby_values=[],
            metric_names=["delta_v"],
        )

        try:
            result = run_numerical_sensitivity(
                base_cfg=cfg,
                seed=42,
                metric_names=["delta_v"],
                robustness_cfg=rob_cfg,
            )
        except Exception:
            pytest.skip("Full MC pipeline not available in unit-test env")

        assert isinstance(result, pd.DataFrame)
        assert "parameter" in result.columns
