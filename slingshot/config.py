"""
Configuration management for slingshot solver.
Supports YAML/JSON config files with Pydantic validation.
"""

from typing import Optional, Literal, Dict, Any, List
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


SelectionMetric = Literal[
    # Barycentric/inertial metrics
    "delta_v",
    "delta_v_pct",
    "delta_v_abs",
    "delta_v_vec",
    "energy_half_dv_vec_sq",
    "deflection",
    "deflection_abs",
    "r_min",
    "r_star_min",
    # Planet-frame diagnostics
    "delta_v_planet_frame",
    "energy_from_planet_orbit",
    # Legacy aliases (kept for backward compatibility)
    "planet_delta_v",
    "bary_delta_v",
    "bary_delta_v_pct",
    "bary_delta_v_abs",
]

SelectionSign = Literal["maximize", "minimize", "abs"]


class SystemConfig(BaseModel):
    """Physical system configuration (star + planet)."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Kepler-432",
            "M_star_Msun": 1.19,
            "M_planet_Mjup": 5.2,
            "R_planet_Rjup": 1.155,
            "R_star_Rsun": 4.06,
            "a_planet_AU": 0.0896,
            "bulk_velocity_vx_kms": 0.0,
            "bulk_velocity_vy_kms": 0.0,
        }
    })
    
    name: str = Field(default="Kepler-432", description="System identifier")
    M_star_Msun: float = Field(default=1.19, ge=0.01, le=10.0, description="Star mass in solar masses")
    M_planet_Mjup: float = Field(default=5.2, ge=0.1, le=100.0, description="Planet mass in Jupiter masses")
    R_planet_Rjup: float = Field(default=1.155, ge=0.1, le=10.0, description="Planet radius in Jupiter radii")
    R_star_Rsun: float = Field(default=4.06, ge=0.1, le=50.0, description="Star radius in solar radii")
    a_planet_AU: float = Field(default=0.0896, ge=0.001, le=1.0, description="Orbital semi-major axis in AU")
    bulk_velocity_vx_kms: float = Field(
        default=0.0,
        ge=-2000.0,
        le=2000.0,
        description="System bulk velocity x-component in km/s (Galilean boost applied to star+planet initial velocities)",
    )
    bulk_velocity_vy_kms: float = Field(
        default=0.0,
        ge=-2000.0,
        le=2000.0,
        description="System bulk velocity y-component in km/s (Galilean boost applied to star+planet initial velocities)",
    )
    
    @field_validator('a_planet_AU')
    @classmethod
    def validate_orbit(cls, v):
        if v <= 0:
            raise ValueError("Orbital semi-major axis must be positive")
        return v


class SamplingConfig(BaseModel):
    """Satellite sampling configuration."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "mode": "barycentric",
            "v_mag_min_kms": 10.0,
            "v_mag_max_kms": 120.0,
            "impact_param_min_AU": 0.5,
            "impact_param_max_AU": 3.0,
            "angle_in_min_deg": -60.0,
            "angle_in_max_deg": 60.0,
        }
    })
    
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
    r_init_AU: Optional[float] = Field(default=None, ge=0.01, description="Fixed initial distance from barycenter (AU). If None, defaults to 2× impact parameter.")
    
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


class NumericalConfig(BaseModel):
    """Numerical integration and analysis parameters."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "rtol": 1e-10,
            "atol": 1e-10,
            "ode_method": "DOP853",
            "r_far_factor": 50.0,
            "min_clearance_factor": 1.05,
            "flyby_r_min_max_hill": 10.0,
            "escape_radius_factor": 3.0,
            "softening_km": 1000.0,
        }
    })
    
    rtol: float = Field(default=1e-10, ge=1e-12, le=1e-6, description="Relative tolerance for ODE solver")
    atol: float = Field(default=1e-10, ge=1e-12, le=1e-6, description="Absolute tolerance for ODE solver")
    
    # ODE solver method
    ode_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = Field(
        default="DOP853",
        description="ODE solver method. DOP853 (order 8) is best for orbital mechanics — "
                    "far fewer steps than RK45 during close encounters. Radau is implicit "
                    "and handles stiff close-encounter regions even better but is slower per step."
    )
    
    # Analysis parameters
    r_far_factor: float = Field(default=50.0, ge=1.0, description="Distance factor for 'far' asymptotic regime (×R_planet)")
    min_clearance_factor: float = Field(default=1.05, ge=1.0, le=2.0, description="Min clearance from planet surface")

    # Flyby completion filter
    flyby_r_min_max_hill: Optional[float] = Field(
        default=10.0, ge=0.1,
        description="Max closest-approach distance for flyby completion, in Hill-sphere radii. "
                    "Particles whose r_min exceeds this are rejected as incomplete flybys. "
                    "Set to None to disable."
    )

    # Star proximity filter
    star_min_clearance_Rstar: Optional[float] = Field(
        default=None, ge=0.0,
        description="Minimum allowed closest approach to the star, in units of R_star. "
                    "Trajectories with r_min_star < star_min_clearance_Rstar × R_star "
                    "are rejected as star-penetrating / star-dominated. "
                    "Set to None to disable (legacy behaviour). "
                    "Recommended: 1.0 (surface collision only) or 2.0–5.0 "
                    "(planet-dominated encounters for slingshot research)."
    )

    # Gravitational softening
    softening_km: float = Field(
        default=1000.0, ge=0.0,
        description="Plummer softening length (km) for satellite–body gravity. "
                    "Replaces 1/r² with 1/(r²+ε²) to cap force gradients during "
                    "close encounters, preventing adaptive-step collapse. "
                    "Only applied to the satellite; the star–planet mutual force "
                    "is never softened. Set to 0 for pure Newtonian gravity. "
                    "A value ≪ R_planet (≈82 573 km) is physically invisible at "
                    "allowed closest-approach distances."
    )

    # Escape termination
    escape_radius_factor: float = Field(
        default=3.0, ge=1.0,
        description="Multiplier on the particle's initial barycentric distance for the escape "
                    "terminal event. Integration stops when the particle recedes to "
                    "escape_radius_factor × r0_bary after periapsis. Higher values let the "
                    "trajectory complete the full swing before termination."
    )


class SelectionObjectiveConfig(BaseModel):
    """Single objective descriptor for Pareto/weighted candidate selection."""

    metric: SelectionMetric = Field(
        default="delta_v",
        description="Objective metric name (supports legacy aliases)",
    )
    sign: SelectionSign = Field(
        default="maximize",
        description="Optimization direction for this objective",
    )
    weight: float = Field(
        default=1.0, gt=0.0,
        description="Relative objective weight (used by weighted mode and Pareto tie-break)",
    )


class PipelineConfig(BaseModel):
    """Monte Carlo pipeline configuration."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "N_particles": 3000,
            "t_mc_max_sec": 1e7,
            "t_best_max_sec": 1e7,
            "n_eval_best": 0,
            "top_frac": 0.10,
            "min_top": 1,
            "select_mode": "single",
            "select_metric": "bary_delta_v_pct",
            "select_sign": "maximize",
            "selection_objectives": [
                {"metric": "bary_delta_v", "sign": "maximize", "weight": 1.0},
                {"metric": "delta_v_vec", "sign": "maximize", "weight": 1.0},
                {"metric": "energy_from_planet_orbit", "sign": "maximize", "weight": 1.0},
            ],
            "weighted_normalization": "minmax",
            "n_parallel": None,
        }
    })
    
    N_particles: int = Field(default=3000, ge=1, le=100000, description="Number of test particles")
    t_mc_max_sec: float = Field(default=1e7, ge=1e4, description="Max integration time for MC sweep (seconds)")
    t_best_max_sec: float = Field(default=1e7, ge=1e4, description="Max integration time for re-run (seconds)")
    n_eval_best: int = Field(default=0, ge=0, description="Output points per best trajectory (0 = adaptive solver steps)")
    top_frac: float = Field(default=0.10, ge=0.01, le=1.0, description="Fraction of successful cases to re-run")
    min_top: int = Field(default=1, ge=1, description="Minimum number of top candidates")
    
    # Selection mode + objectives
    select_mode: Literal["single", "pareto", "weighted"] = Field(
        default="single",
        description="Candidate selection strategy: single-objective, Pareto front, or weighted multi-objective",
    )
    select_metric: SelectionMetric = Field(
        default="bary_delta_v_pct",
        description="Single-mode objective metric (also supports legacy alias names)",
    )
    select_sign: SelectionSign = Field(
        default="maximize",
        description="Single-mode optimization direction: maximize, minimize, or abs",
    )
    selection_objectives: List[SelectionObjectiveConfig] = Field(
        default_factory=lambda: [
            SelectionObjectiveConfig(metric="bary_delta_v", sign="maximize", weight=1.0),
            SelectionObjectiveConfig(metric="delta_v_vec", sign="maximize", weight=1.0),
            SelectionObjectiveConfig(metric="energy_from_planet_orbit", sign="maximize", weight=1.0),
        ],
        description="Objectives used by pareto/weighted modes",
    )
    weighted_normalization: Literal["minmax", "rank"] = Field(
        default="minmax",
        description="Normalization method for weighted-mode scalarization and Pareto tie-breaks",
    )
    
    # Reproducibility
    seed: Optional[int] = Field(
        default=None, ge=0,
        description="Random seed for Monte Carlo sampling. None = non-deterministic. "
                    "Set to a fixed integer (e.g. 42) for reproducible paper runs.",
    )

    # Parallelization
    n_parallel: Optional[int] = Field(default=None, ge=1, description="Number of parallel workers (None = auto-detect)")

    @field_validator("selection_objectives")
    @classmethod
    def validate_selection_objectives(cls, v: List[SelectionObjectiveConfig]):
        if len(v) == 0:
            raise ValueError("selection_objectives must contain at least one objective")
        return v


class VisualizationConfig(BaseModel):
    """Visualization and animation configuration."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "render_video": True,
            "video_fps": 30,
            "video_format": "mp4",
            "animate_trajectory": True,
            "animate_phase_space": True,
            "animate_comparison": False,
            "figure_dpi": 150,
            "figure_format": "png",
            "generate_publication_dashboard": True,
            "generate_candidate_ranking_plot": True,
            "split_subplot_figures": True,
            "generate_poincare_maps": True,
            "generate_oberth_maps": False,
            "heatmap_grid_resolution": 60,
            "heatmap_approach_angles_deg": [0.0, 45.0, 85.0],
            "top_n_overlay": 5,
            "trajectory_gradient_mode": "hexbin",
            "trajectory_overlay_lines": True,
            "trajectory_overlay_line_count": 90,
            "trajectory_confidence_min_count": 2,
            "trajectory_energy_norm_mode": "auto",
            "trajectory_energy_vmin": None,
            "trajectory_energy_vmax": None,
            "trajectory_hexbin_gridsize": 150,
            "trajectory_kde_sigma_bins": 2.0,
            "trajectory_time_frames": 48,
            "trajectory_time_export_npz": True,
        }
    })
    
    render_video: bool = Field(default=True, description="Generate animation frames and videos")
    video_fps: int = Field(default=30, ge=1, le=120, description="Video frames per second")
    video_format: Literal["mp4", "gif"] = Field(default="mp4", description="Output video format")
    
    # Animation types
    animate_trajectory: bool = Field(default=True, description="Animate single trajectory with bodies")
    animate_phase_space: bool = Field(default=True, description="Animate phase-space evolution")
    animate_comparison: bool = Field(default=False, description="Animate multi-trajectory comparisons")
    
    figure_dpi: int = Field(default=150, ge=50, le=600, description="Figure resolution (DPI)")
    figure_format: Literal["png", "pdf", "jpg"] = Field(default="png", description="Diagnostic plot format")

    # Publication diagnostics
    generate_publication_dashboard: bool = Field(
        default=True,
        description="Generate publication-oriented dashboard that summarizes objective evidence",
    )
    generate_candidate_ranking_plot: bool = Field(
        default=True,
        description="Generate detailed ranking diagnostics for selected re-run candidates",
    )
    split_subplot_figures: bool = Field(
        default=True,
        description="Also export each subplot panel as an individual image for report-ready records",
    )

    # 2-body diagnostic plot toggles
    generate_2body_heatmaps: bool = Field(
        default=False,
        description="Deprecated toggle (2D encounter maps removed from generation pipeline)"
    )
    generate_scattering_maps: bool = Field(
        default=False,
        description="Deprecated toggle (scattering maps removed from generation pipeline)"
    )
    generate_poincare_maps: bool = Field(
        default=True,
        description="Generate Poincaré-style (b vs α_inf) parameter-space contourf maps"
    )
    generate_oberth_maps: bool = Field(
        default=False,
        description="Generate Oberth burn comparison maps (no-burn vs burn gain)"
    )
    generate_trajectory_heatmap: bool = Field(
        default=False,
        description="Generate spatial energy density / flux heatmap (slow: 150×200 grid binned to 500×500)"
    )

    # 2-body grid parameters
    heatmap_grid_resolution: int = Field(
        default=60, ge=10, le=200,
        description="Grid resolution per axis for 2-body heatmaps (N×N)"
    )
    heatmap_approach_angles_deg: list[float] = Field(
        default=[0.0, 45.0, 85.0],
        description="Approach angles (degrees) for multi-scenario 2-body diagnostics"
    )

    # Multi-candidate overlay
    top_n_overlay: int = Field(
        default=5, ge=1, le=20,
        description="Number of top candidates to overlay on multi-trajectory plot"
    )

    # Trajectory-gradient renderer controls (plotting_twobody.plot_trajectory_tracks)
    trajectory_gradient_mode: Literal["legacy", "line_overlay", "hexbin", "kde", "time_video"] = Field(
        default="hexbin",
        description="Trajectory rendering mode for narrowed trajectory tracks",
    )
    trajectory_overlay_lines: bool = Field(
        default=True,
        description="Overlay representative trajectory lines on top of gradient fields",
    )
    trajectory_overlay_line_count: int = Field(
        default=90, ge=0, le=2000,
        description="Maximum number of representative trajectory lines to draw when overlay is enabled",
    )
    trajectory_confidence_min_count: int = Field(
        default=2, ge=0, le=10000,
        description="Minimum samples per bin before a gradient pixel is considered valid",
    )
    trajectory_energy_norm_mode: Literal["auto", "fixed"] = Field(
        default="auto",
        description="Energy colormap normalization mode for trajectory/phase gradients",
    )
    trajectory_energy_vmin: Optional[float] = Field(
        default=None,
        description="Fixed normalization minimum for scattering energy (used when norm_mode='fixed')",
    )
    trajectory_energy_vmax: Optional[float] = Field(
        default=None,
        description="Fixed normalization maximum for scattering energy (used when norm_mode='fixed')",
    )
    trajectory_hexbin_gridsize: int = Field(
        default=150, ge=20, le=600,
        description="Hexbin resolution for trajectory gradient when mode='hexbin'",
    )
    trajectory_kde_sigma_bins: float = Field(
        default=2.0, ge=0.1, le=20.0,
        description="Gaussian smoothing width in bins for mode='kde' or 'time_video'",
    )
    trajectory_time_frames: int = Field(
        default=48, ge=2, le=500,
        description="Number of frames exported for mode='time_video'",
    )
    trajectory_time_export_npz: bool = Field(
        default=True,
        description="Export time-evolution NPZ cubes when mode='time_video'",
    )

    @model_validator(mode="after")
    def validate_trajectory_energy_norm(self):
        # Hard-disable deprecated 2D/scattering map generation toggles.
        self.generate_2body_heatmaps = False
        self.generate_scattering_maps = False

        if self.trajectory_energy_norm_mode == "fixed":
            if self.trajectory_energy_vmin is None or self.trajectory_energy_vmax is None:
                raise ValueError(
                    "trajectory_energy_vmin and trajectory_energy_vmax must be set when "
                    "trajectory_energy_norm_mode='fixed'"
                )
            if self.trajectory_energy_vmax <= self.trajectory_energy_vmin:
                raise ValueError(
                    "trajectory_energy_vmax must be greater than trajectory_energy_vmin"
                )
        return self


class TwoBodyConfig(BaseModel):
    """Two-body encounter scan configuration (consumed by trajectory_tracks.py)."""

    model_config = ConfigDict(extra="ignore")

    scattering_body: Literal["star", "planet", "both"] = Field(
        default="both",
        description="Which body to scatter off: 'star', 'planet', or 'both'",
    )

    # Baseline mode
    baseline_mode: Literal["fixed", "narrowed"] = Field(
        default="narrowed",
        description="'fixed' uses config v_approach; 'narrowed' derives envelope from 3-body top candidates",
    )
    padding_factor: float = Field(
        default=1.5, ge=1.0, le=5.0,
        description="Multiplicative padding on narrowed envelope edges",
    )
    num_v: int = Field(default=20, ge=1, description="Velocity grid points for narrowed sweep")
    num_b_narrow: int = Field(default=100, ge=1, description="Impact-param grid for narrowed sweep")
    num_angles_narrow: int = Field(default=100, ge=1, description="Angle grid for narrowed sweep")

    # Approach geometry  [km-kg-s]
    v_approach_kms: float = Field(default=50.0, ge=0.1, description="Approach velocity [km/s]")
    vstar0_kms: float = Field(
        default=10.0, ge=0.0,
        description="Star speed parameter for standalone/fixed 2-body scans [km/s]. "
                    "In pipeline comparisons, this may be auto-synced to the 3-body initial star speed."
    )
    r_start_km: float = Field(default=1.0e11, ge=1.0, description="Initial separation [km]")

    # Impact-parameter scan  [km]
    b_min_km: float = Field(default=1.0e7, ge=1.0, description="Min impact parameter [km]")
    b_max_km: float = Field(default=4.0e9, ge=1.0, description="Max impact parameter [km]")
    num_b: int = Field(default=150, ge=1, description="Number of impact-parameter samples")
    log_spacing: bool = Field(default=True, description="Use log spacing for b")

    # Angle scan
    angle_min_deg: float = Field(default=270.0, description="Min approach angle [deg]")
    angle_max_deg: float = Field(default=360.0, description="Max approach angle [deg]")
    num_angles: int = Field(default=200, ge=1, description="Number of angle samples")

    # Integration
    num_points: int = Field(default=400, ge=10, description="Points per trajectory")
    output_dir: str = Field(default="results/two_body", description="Output directory")


class ParameterDistConfig(BaseModel):
    """Single parameter posterior distribution (Gaussian)."""
    mean: float = Field(description="Posterior mean")
    std: float = Field(ge=0.0, description="Posterior standard deviation")


class UncertaintyConfig(BaseModel):
    """Configuration for parameter-posterior uncertainty propagation."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable uncertainty propagation analysis",
    )
    n_draws: int = Field(
        default=50, ge=1, le=10000,
        description="Number of posterior parameter draws",
    )
    seed: Optional[int] = Field(
        default=None, ge=0,
        description="Random seed for posterior sampling (None = non-deterministic)",
    )
    bootstrap_n_resample: int = Field(
        default=200, ge=10, le=10000,
        description="Number of bootstrap resamples for Pareto stability analysis",
    )
    parameters: Dict[str, ParameterDistConfig] = Field(
        default_factory=dict,
        description="Parameter posteriors keyed by SystemConfig field name, "
                    "e.g. {'M_planet_Mjup': {'mean': 5.2, 'std': 0.4}}",
    )


class RobustnessConfig(BaseModel):
    """Configuration for numerical robustness / sensitivity analysis."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable robustness / sensitivity analysis",
    )
    seed: Optional[int] = Field(
        default=None, ge=0,
        description="Random seed for robustness runs (None = use pipeline seed)",
    )
    convergence_N: List[int] = Field(
        default=[500, 1000, 2000, 4000, 8000],
        description="Particle counts for N-convergence test",
    )
    softening_values: List[float] = Field(
        default=[0.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
        description="Softening lengths (km) to sweep",
    )
    tolerance_values: List[float] = Field(
        default=[1e-10, 1e-11, 1e-12, 1e-13],
        description="Solver tolerance values to sweep (applied to both rtol and atol)",
    )
    clearance_values: List[float] = Field(
        default=[1.5, 2.0, 3.0, 5.0],
        description="Star min clearance (R_star) values to sweep",
    )
    flyby_values: List[float] = Field(
        default=[0.3, 0.5, 0.8, 1.0, 3.0, 5.0],
        description="Flyby r_min (Hill radii) values to sweep",
    )
    metric_names: List[str] = Field(
        default=["delta_v", "delta_v_vec", "energy_from_planet_orbit"],
        description="Metrics to track across sensitivity sweeps",
    )


class TieringConfig(BaseModel):
    """Physics-based candidate tiering thresholds.

    Candidates are classified by the ratio ε_planet / |Δε_monopole|:
      - planet-dominated  (ratio > planet_dominated_threshold)
      - hybrid            (hybrid_threshold ≤ ratio ≤ planet_dominated_threshold)
      - star-dominated    (ratio < hybrid_threshold)
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True, description="Enable tier classification in report / notebook")
    planet_dominated_threshold: float = Field(default=2.0, description="Min ratio for planet-dominated tier")
    hybrid_threshold: float = Field(default=0.5, description="Min ratio for hybrid tier")
    top_n_per_tier: int = Field(default=10, description="Max candidates shown per tier in report tables")


class FullConfig(BaseModel):
    """Complete configuration for slingshot solver pipeline."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"description": "Complete configuration for slingshot-solver"},
    )

    system: SystemConfig = Field(default_factory=SystemConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    two_body: Optional[TwoBodyConfig] = Field(default=None, description="2-body scan config")
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig, description="Uncertainty propagation config")
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig, description="Robustness / sensitivity analysis config")
    tiering: TieringConfig = Field(default_factory=TieringConfig, description="Physics-based candidate tiering config")


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
            if yaml is None:
                raise ImportError("pyyaml is required for YAML config files. Install with: pip install pyyaml")
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
            if yaml is None:
                raise ImportError("pyyaml is required for YAML config files. Install with: pip install pyyaml")
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format.lower() == 'json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
