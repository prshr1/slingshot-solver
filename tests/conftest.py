"""
Shared test fixtures for slingshot-solver.
"""

import pytest
import numpy as np


@pytest.fixture
def default_config_path():
    """Path to the default config YAML."""
    from pathlib import Path
    return Path(__file__).parent.parent / "configs" / "config_default.yaml"


@pytest.fixture
def kepler432_config_path():
    """Path to the Kepler-432 config YAML."""
    from pathlib import Path
    return Path(__file__).parent.parent / "configs" / "config_kepler432_case.yaml"


@pytest.fixture
def sample_star_planet_state():
    """Barycentric star+planet initial state for testing."""
    from slingshot.core.dynamics import init_hot_jupiter_barycentric
    return init_hot_jupiter_barycentric()


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)
