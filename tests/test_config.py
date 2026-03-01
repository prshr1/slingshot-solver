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
