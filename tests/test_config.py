"""
Tests for configuration loading and validation.
"""

import pytest
from pathlib import Path


class TestConfigLoad:
    """Test config YAML loading and Pydantic validation."""

    def test_load_default_config(self, default_config_path):
        from slingshot.config import load_config
        cfg = load_config(str(default_config_path))
        assert cfg is not None
        assert cfg.system.name is not None
        assert cfg.pipeline.N_particles > 0

    def test_load_kepler432_config(self, kepler432_config_path):
        from slingshot.config import load_config
        if not kepler432_config_path.exists():
            pytest.skip("Kepler-432 config not available")
        cfg = load_config(str(kepler432_config_path))
        assert cfg.system.M_star_Msun > 0
        assert cfg.system.M_planet_Mjup > 0

    def test_roundtrip_save_load(self, default_config_path, tmp_path):
        from slingshot.config import load_config, save_config
        cfg = load_config(str(default_config_path))
        out = tmp_path / "test_config.yaml"
        save_config(cfg, str(out))
        cfg2 = load_config(str(out))
        assert cfg.system.name == cfg2.system.name
        assert cfg.pipeline.N_particles == cfg2.pipeline.N_particles

    def test_system_presets(self):
        from slingshot.config import load_system_config
        sys_cfg = load_system_config("kepler-432")
        assert sys_cfg.M_star_Msun > 0


class TestConfigModels:
    """Test individual Pydantic config models."""

    def test_system_config_defaults(self):
        from slingshot.config import SystemConfig
        sys = SystemConfig(name="test", M_star_Msun=1.0, M_planet_Mjup=1.0)
        assert sys.M_star_Msun == 1.0

    def test_full_config_requires_system(self):
        from slingshot.config import FullConfig, SystemConfig
        sys = SystemConfig(name="test", M_star_Msun=1.0, M_planet_Mjup=1.0)
        cfg = FullConfig(system=sys)
        assert cfg.system.name == "test"


class TestPipelineSeed:
    """Test seed field on PipelineConfig."""

    def test_seed_none_by_default(self):
        from slingshot.config import PipelineConfig
        p = PipelineConfig()
        assert p.seed is None

    def test_seed_accepts_int(self):
        from slingshot.config import PipelineConfig
        p = PipelineConfig(seed=42)
        assert p.seed == 42

    def test_seed_rejects_negative(self):
        from slingshot.config import PipelineConfig
        with pytest.raises(Exception):
            PipelineConfig(seed=-1)


class TestUncertaintyConfig:
    """Test UncertaintyConfig model."""

    def test_defaults(self):
        from slingshot.config import UncertaintyConfig
        u = UncertaintyConfig()
        assert u.enabled is False
        assert u.n_draws > 0

    def test_enabled_with_params(self):
        from slingshot.config import UncertaintyConfig, ParameterDistConfig
        u = UncertaintyConfig(
            enabled=True,
            n_draws=100,
            seed=99,
            parameters={"M_planet_Mjup": ParameterDistConfig(mean=5.2, std=0.4)},
        )
        assert u.enabled is True
        assert "M_planet_Mjup" in u.parameters


class TestRobustnessConfig:
    """Test RobustnessConfig model."""

    def test_defaults(self):
        from slingshot.config import RobustnessConfig
        r = RobustnessConfig()
        assert r.enabled is False

    def test_custom_sweep(self):
        from slingshot.config import RobustnessConfig
        r = RobustnessConfig(
            enabled=True,
            softening_values=[1e3, 1e4, 1e5],
        )
        assert len(r.softening_values) == 3


class TestNewSectionsInYAML:
    """Test that YAML configs with new uncertainty/robustness sections load."""

    def test_interstellar_yaml_loads_new_sections(self):
        from pathlib import Path
        from slingshot.config import load_config

        p = Path(__file__).parent.parent / "configs" / "config_interstellar_k432.yaml"
        if not p.exists():
            pytest.skip("interstellar config not available")
        cfg = load_config(str(p))
        assert cfg.uncertainty is not None
        assert cfg.uncertainty.enabled is True
        assert len(cfg.uncertainty.parameters) > 0

    def test_default_yaml_no_uncertainty(self, default_config_path):
        from slingshot.config import load_config
        cfg = load_config(str(default_config_path))
        # Should either be None or disabled
        if cfg.uncertainty is not None:
            assert cfg.uncertainty.enabled is False
