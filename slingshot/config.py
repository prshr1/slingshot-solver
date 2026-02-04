"""
Configuration management for slingshot solver.
Supports YAML/JSON config files with Pydantic validation.
"""

from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
import json
import yaml
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for users without pydantic
    BaseModel = object
    Field = lambda **kwargs: None


# Physical Constants (SI units or as specified)
@dataclass
class PhysicalConstants:
    """Physical constants used in simulations."""
    G: float = 6.67430e-20  # km^3 / (kg s^2)
    M_SUN: float = 1.98847e30  # kg
    M_JUP: float = 1.898e27  # kg
    R_JUP: float = 71492.0  # km
    AU_KM: float = 1.495978707e8  # km
    R_SUN: float = 696000.0  # km


class SystemConfig(BaseModel if BaseModel != object else object):
    """Physical system configuration (star + planet)."""
    
    name: str = Field(default="Kepler-432", description="System identifier")
    M_star_Msun: float = Field(default=1.19, ge=0.01, le=10.0, description="Star mass in solar masses")
    M_planet_Mjup: float = Field(default=5.2, ge=0.1, le=100.0, description="Planet mass in Jupiter masses")
    R_planet_Rjup: float = Field(default=1.155, ge=0.1, le=10.0, description="Planet radius in Jupiter radii")
    a_planet_AU: float = Field(default=0.0896, ge=0.001, le=1.0, description="Orbital semi-major axis in AU")
    
    if BaseModel != object:
        @validator('a_planet_AU')
        def validate_orbit(cls, v, values):
            if v <= 0:
                raise ValueError("Orbital semi-major axis must be positive")
            return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Kepler-432",
                "M_star_Msun": 1.19,
                "M_planet_Mjup": 5.2,
                "R_planet_Rjup": 1.155,
                "a_planet_AU": 0.0896,
            }
        }


class SamplingConfig(BaseModel if BaseModel != object else object):
    """Satellite sampling configuration."""
    
    mode: Literal["barycentric", "planet"] = Field(
        default="barycentric",
        description="Sampling mode: 'barycentric' (parametric hyperbolic) or 'planet' (orbit-relative)"
    )
    
    # Barycentric mode parameters
    v_mag_min_kms: float = Field(default=10.0, ge=0.1, description="Min velocity magnitude (km/s)")
    v_mag_max_kms: float = Field(default=120.0, ge=0.1, description="Max velocity magnitude (km/s)")
    impact_param_min_AU: float = Field(default=0.5, ge=0.01, description="Min impact parameter (AU)")
    impact_param_max_AU: float = Field(default=3.0, ge=0.01, description="Max impact parameter (AU)")
    angle_in_min_deg: float = Field(default=-60.0, ge=-180.0, le=180.0, description="Min incoming angle (deg)")
    angle_in_max_deg: float = Field(default=60.0, ge=-180.0, le=180.0, description="Max incoming angle (deg)")
    
    # Planet-frame mode parameters
    r_min_factor: Optional[float] = Field(default=20.0, ge=1.0, description="Min radius in R_planet units")
    r_max_factor: Optional[float] = Field(default=500.0, ge=1.0, description="Max radius in R_planet units")
    v_rel_min_kms: Optional[float] = Field(default=12.0, ge=0.1, description="Min relative velocity (km/s)")
    v_rel_max_kms: Optional[float] = Field(default=80.0, ge=0.1, description="Max relative velocity (km/s)")
    
    # Barycentric filtering
    bary_unbound_requirement: Optional[Literal["pre", "post", "either", "both"]] = Field(
        default="both",
        description="Require barycentric escape at: 'pre' (initial), 'post' (final), 'either', 'both', or None"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "mode": "barycentric",
                "v_mag_min_kms": 10.0,
                "v_mag_max_kms": 120.0,
                "impact_param_min_AU": 0.5,
                "impact_param_max_AU": 3.0,
                "angle_in_min_deg": -60.0,
                "angle_in_max_deg": 60.0,
            }
        }


class NumericalConfig(BaseModel if BaseModel != object else object):
    """Numerical integration and analysis parameters."""
    
    rtol: float = Field(default=1e-10, ge=1e-12, le=1e-6, description="Relative tolerance for ODE solver")
    atol: float = Field(default=1e-10, ge=1e-12, le=1e-6, description="Absolute tolerance for ODE solver")
    
    # Analysis parameters
    r_far_factor: float = Field(default=20.0, ge=1.0, description="Distance factor for 'far' asymptotic regime")
    min_clearance_factor: float = Field(default=1.05, ge=1.0, le=2.0, description="Min clearance from planet surface")
    
    class Config:
        schema_extra = {
            "example": {
                "rtol": 1e-10,
                "atol": 1e-10,
                "r_far_factor": 20.0,
                "min_clearance_factor": 1.05,
            }
        }


class PipelineConfig(BaseModel if BaseModel != object else object):
    """Monte Carlo pipeline configuration."""
    
    N_particles: int = Field(default=3000, ge=1, le=100000, description="Number of test particles")
    t_mc_max_sec: float = Field(default=1e7, ge=1e4, description="Max integration time for MC sweep (seconds)")
    t_best_max_sec: float = Field(default=1e7, ge=1e4, description="Max integration time for re-run (seconds)")
    n_eval_best: int = Field(default=1000, ge=10, description="Output points per best candidate trajectory")
    top_frac: float = Field(default=0.10, ge=0.01, le=1.0, description="Fraction of successful cases to re-run")
    min_top: int = Field(default=1, ge=1, description="Minimum number of top candidates")
    
    # Selection metric
    select_metric: Literal["planet_delta_v", "bary_delta_v", "bary_delta_v_pct", "bary_delta_v_abs"] = Field(
        default="bary_delta_v_pct",
        description="Metric to optimize when selecting candidates"
    )
    select_sign: Literal["maximize", "minimize", "abs"] = Field(
        default="maximize",
        description="Optimization direction: maximize, minimize, or abs (largest absolute value)"
    )
    
    # Parallelization
    n_parallel: Optional[int] = Field(default=None, ge=1, description="Number of parallel workers (None = auto-detect)")
    
    class Config:
        schema_extra = {
            "example": {
                "N_particles": 3000,
                "t_mc_max_sec": 1e7,
                "t_best_max_sec": 1e7,
                "n_eval_best": 1000,
                "top_frac": 0.10,
                "min_top": 1,
                "select_metric": "bary_delta_v_pct",
                "select_sign": "maximize",
                "n_parallel": None,
            }
        }


class VisualizationConfig(BaseModel if BaseModel != object else object):
    """Visualization and animation configuration."""
    
    render_video: bool = Field(default=True, description="Generate animation frames and videos")
    video_fps: int = Field(default=30, ge=1, le=120, description="Video frames per second")
    video_format: Literal["mp4", "gif"] = Field(default="mp4", description="Output video format")
    
    # Animation types
    animate_trajectory: bool = Field(default=True, description="Animate single trajectory with bodies")
    animate_phase_space: bool = Field(default=True, description="Animate phase-space evolution")
    animate_comparison: bool = Field(default=False, description="Animate multi-trajectory comparisons")
    
    figure_dpi: int = Field(default=100, ge=50, le=300, description="Figure resolution (DPI)")
    figure_format: Literal["png", "pdf", "jpg"] = Field(default="png", description="Diagnostic plot format")
    
    class Config:
        schema_extra = {
            "example": {
                "render_video": True,
                "video_fps": 30,
                "video_format": "mp4",
                "animate_trajectory": True,
                "animate_phase_space": True,
                "animate_comparison": False,
                "figure_dpi": 100,
                "figure_format": "png",
            }
        }


class FullConfig(BaseModel if BaseModel != object else object):
    """Complete configuration for slingshot solver pipeline."""
    
    system: SystemConfig = Field(default_factory=SystemConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    
    class Config:
        schema_extra = {
            "description": "Complete configuration for slingshot-solver"
        }


def load_config(config_path: str) -> FullConfig:
    """
    Load configuration from YAML or JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML or JSON config file
    
    Returns
    -------
    FullConfig
        Validated configuration object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    if data is None:
        data = {}
    
    return FullConfig(**data)


def load_system_config(system_name: str) -> SystemConfig:
    """
    Load a predefined system configuration by name.
    
    Parameters
    ----------
    system_name : str
        System identifier: 'kepler-432', 'toi-1431', etc.
    
    Returns
    -------
    SystemConfig
        System configuration
    """
    systems = {
        'kepler-432': SystemConfig(
            name="Kepler-432",
            M_star_Msun=1.19,
            M_planet_Mjup=5.2,
            R_planet_Rjup=1.155,
            a_planet_AU=0.0896,
        ),
        'toi-1431': SystemConfig(
            name="TOI-1431",
            M_star_Msun=1.9,
            M_planet_Mjup=3.12,
            R_planet_Rjup=1.49,
            a_planet_AU=0.047,
        ),
    }
    
    key = system_name.lower()
    if key not in systems:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(systems.keys())}")
    
    return systems[key]


def save_config(config: FullConfig, output_path: str, format: Literal["yaml", "json"] = "yaml") -> None:
    """
    Save configuration to file.
    
    Parameters
    ----------
    config : FullConfig
        Configuration object to save
    output_path : str
        Path to output file
    format : str
        Output format: 'yaml' or 'json'
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.dict()
    
    with open(path, 'w') as f:
        if format.lower() in ['yaml', 'yml']:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
