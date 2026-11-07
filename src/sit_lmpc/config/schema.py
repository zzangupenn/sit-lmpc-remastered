"""Immutable configuration: schema fallbacks, selected YAML, then CLI overrides."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from importlib.resources import files
import math
import yaml


@dataclass(frozen=True)
class SimulationConfig:
    model: str = "dynamic_ST"
    map: str = "custom1"
    timestep: float = 0.1
    integrator_timestep: float = 0.02
    friction: float = 0.6
    mass_ratio: float = 1.0
    initial_speed: float = 7.0
    max_laps: int = 8
    max_steps: int = 100000
    seed_mode: str = "auto"
    seed: int | None = None
    dtype: str = "float32"
    action_scale: tuple[float, float] = (0.4, 3.5)
    control_noise: tuple[float, float] = (0.001, 0.001)
    state_noise: float = 0.001


@dataclass(frozen=True)
class ModelBankConfig:
    frictions: tuple[float, ...] = (0.3, 0.4, 0.7, 0.8)


@dataclass(frozen=True)
class BeliefConfig:
    enabled: bool = False
    backend: str = "model_bank"
    predictor: str = "ensemble"
    prior: tuple[float, ...] | None = None
    history_samples: int = 4096
    recent_fraction: float = 0.5
    evidence_strength: float = 32.0
    residual_scales: tuple[float, ...] = (0.05, 0.05, 0.01, 0.1, 0.01, 0.1, 0.01)
    max_standardized_rmse: float = 5.0
    batch_size: int = 512
    fit_horizon: int | None = None
    fit_max_iterations: int = 100
    fit_tolerance: float = 1e-6
    mean_fit_slack: float = 1e-4

    def __post_init__(self):
        # Accept the user's spelling without breaking existing saved ensemble_fit configs.
        if self.backend == "fit_ensemble":
            object.__setattr__(self, "backend", "ensemble_fit")


@dataclass(frozen=True)
class RiskConstraintConfig:
    enabled: bool = False
    mode: str = "terminal"
    evaluation: str = "predictor"
    lambda_count: int = 16
    lambda_max: float = 20.0
    max_candidates: int = 256


@dataclass(frozen=True)
class ProbabilisticSafetyConfig:
    enabled: bool = False
    boundary_threshold: float = 0.05
    terminal_hull_threshold: float = 0.05
    lambda_count: int = 16
    lambda_max: float = 20.0
    max_candidates: int = 256
    fallback: str = "baseline"


@dataclass(frozen=True)
class DynamicsConfig:
    backend: str = "st"
    checkpoint: str | None = None
    particles: int = 64
    verification_particles: int = 2048
    verification_confidence: float = 0.95
    belief_update: str = "frozen"
    likelihood_power: float = 1.0
    update_batch_size: int = 512
    validation_fraction: float = 0.2
    validation_particles: int = 64
    validation_max_windows: int = 64
    chance_estimation: str = "independent"
    posterior_method: str = "tempered"
    calibration_fraction: float = 0.2
    calibration_split: str = "random_windows"
    calibration_horizon: int = 10
    calibration_min_windows: int = 4
    calibration_target_coverage: float = 0.95
    calibration_score_slack: float = 0.05
    calibration_powers: tuple[float, ...] = (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)

    def __post_init__(self):
        if isinstance(self.calibration_powers, list):
            object.__setattr__(self, "calibration_powers", tuple(self.calibration_powers))


@dataclass(frozen=True)
class SafeSetDensityConfig:
    method: str = "quantile"
    hidden_dims: int = 64
    layers: int = 4
    bins: int = 8
    epochs: int = 50
    min_training_batches: int = 0
    steps: int | None = None
    batch_size: int = 256
    learning_rate: float = 1e-3
    min_training_states: int = 128
    min_calibration_states: int = 64
    max_states_per_partition: int = 20000  # Legacy config compatibility; NF uses all retained states.
    progress_bins: int = 8
    calibration: str = "global"
    relaxation_quantile: float = 0.99
    penalty_upperbound_quantile: float = 0.999
    sigma_cutoff_quantile: float = 0.20
    penalty_sigma: float = 4.0
    penalty_transform: str = "linear"
    penalty_normalization: str = "calibrated"


@dataclass(frozen=True)
class PlannerConfig:
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    sit_lmpc_mode: bool = False
    friction: float = 0.6
    mass_ratio: float = 1.0
    model_bank: ModelBankConfig = field(default_factory=ModelBankConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    risk_constraint: RiskConstraintConfig = field(default_factory=RiskConstraintConfig)
    probabilistic_safety: ProbabilisticSafetyConfig = field(default_factory=ProbabilisticSafetyConfig)
    horizon: int = 10
    samples: int = 1024
    iterations: int = 1
    sample_std: tuple[float, float] = (0.25, 0.25)
    temperature: float = 0.01
    damping: float = 0.001
    half_width: float = 4.0
    boundary_buffer: float = 0.1
    lambda_count: int = 16
    lambda_max: float = 20.0
    safe_set_size: int = 6
    safe_set_min_states: int = 0
    safe_set_max_points: int = 2000000
    hull_selection_points: int = 1000
    safe_set_query: str = "facets"
    safe_set_density: SafeSetDensityConfig = field(default_factory=SafeSetDensityConfig)
    implicit_hull_point_bucket: int = 64
    implicit_hull_max_iterations: int = 128
    implicit_hull_tolerance: float = 1e-4
    implicit_hull_block_size: int = 1024
    hull_bucket: int = 10
    hull_precompile: bool = True
    hull_precompile_limit: int = 1200
    safe_set_relaxation: float = 0.03
    safe_set_reward_limit: float = 1.0
    extension_seconds: float = 2.0
    reference_interval: float = 3.0
    search_points: int = 400
    action_rate_penalty: float = 0.01
    mean_value: bool = True
    scan_rollout: bool = True
    safe_set_selection: str = "exact"
    safe_set_window_margin: float = 5.0
    projection_block_size: int = 8192

    @property
    def boundary_limit(self):
        return self.half_width - self.boundary_buffer

    @property
    def safe_set_threshold(self):
        # NF check violations are measured beyond their own calibrated NLL bound.
        return 0.0 if self.safe_set_query == "nf_density" else self.safe_set_relaxation


@dataclass(frozen=True)
class ValueConfig:
    model: str = "normalizing_flow"
    input_mode: str = "state"
    hidden_dims: int = 128
    layers: int = 4
    bins: int = 8
    activation: str = "swish"
    batch_size: int = 300
    epochs: int = 300
    learning_rate: float = 5e-5
    samples: int = 20
    boundary_samples: int = 100


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True
    input_mode: str = "state"
    buffer_size: int | None = None
    failure_fraction: float = 0.5
    validation_fraction: float = 0.2
    cutoff_mode: str = "learned"
    manual_cutoff: float = 0.5
    target_safe_fraction: float = 0.95
    cutoff_min_states: int = 32
    cutoff_min_episodes: int = 2
    hidden_dims: int = 64
    layers: int = 2
    batch_size: int = 256
    batches_per_epoch: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class DiagnosticsConfig:
    render: bool = True
    plots: bool = False
    plot_interval: int = 10
    trace_steps: int = 50
    keep_open_on_failure: bool = False
    summary_plots: bool = True
    keyboard_controls: bool = False
    print_lambdas: bool = False


@dataclass(frozen=True)
class OutputConfig:
    root: str = "results"
    exp_name: str = "st"
    directory: str | None = None
    resume: str | None = None
    checkpoint_every_trials: int = 10
    quick_load: bool = False
    quick_load_init: bool = True
    init_path: str | None = None
    trust_legacy_init: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def validate(self):
        s, p, v, d = self.simulation, self.planner, self.value, self.diagnostics
        r = self.risk
        if not isinstance(d.print_lambdas, bool):
            raise ValueError("diagnostics.print_lambdas must be true or false")
        dynamics = p.dynamics
        if not isinstance(dynamics, DynamicsConfig) or dynamics.backend not in ("st", "gaussian_bnn"):
            raise ValueError("planner.dynamics.backend must be st or gaussian_bnn")
        if dynamics.checkpoint is not None and (not isinstance(dynamics.checkpoint, str) or not dynamics.checkpoint.strip()):
            raise ValueError("planner.dynamics.checkpoint must be null or a nonempty path")
        for name in ("particles", "verification_particles", "update_batch_size",
                     "validation_particles", "validation_max_windows"):
            if type(getattr(dynamics, name)) is not int or getattr(dynamics, name) < 1:
                raise ValueError(f"planner.dynamics.{name} must be a positive integer")
        if (isinstance(dynamics.verification_confidence, bool)
                or not isinstance(dynamics.verification_confidence, (int, float))
                or not 0 < dynamics.verification_confidence < 1):
            raise ValueError("planner.dynamics.verification_confidence must be in (0, 1)")
        if dynamics.chance_estimation not in ("independent", "planning_samples"):
            raise ValueError("planner.dynamics.chance_estimation must be independent or planning_samples")
        if dynamics.backend == "gaussian_bnn":
            if p.sit_lmpc_mode or p.belief.enabled:
                raise ValueError("gaussian_bnn requires sit_lmpc_mode=false and belief.enabled=false; use dynamics.belief_update=frozen or posterior")
        if dynamics.belief_update not in ("frozen", "posterior"):
            raise ValueError("planner.dynamics.belief_update must be frozen or posterior")
        if dynamics.belief_update != "frozen" and dynamics.backend != "gaussian_bnn":
            raise ValueError("posterior belief_update requires gaussian_bnn dynamics")
        if dynamics.posterior_method not in ("standard", "tempered", "noise", "noise_tempered"):
            raise ValueError("planner.dynamics.posterior_method must be standard, tempered, noise or noise_tempered")
        if dynamics.calibration_split not in ("random_windows", "contiguous"):
            raise ValueError("planner.dynamics.calibration_split must be random_windows or contiguous")
        for name in ("calibration_horizon", "calibration_min_windows"):
            if type(getattr(dynamics, name)) is not int or getattr(dynamics, name) < 1:
                raise ValueError(f"planner.dynamics.{name} must be a positive integer")
        for name, upper, inclusive in (("calibration_fraction", .5, False),
                                       ("calibration_target_coverage", 1, True),
                                       ("calibration_score_slack", 1, True)):
            value = getattr(dynamics, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (value > upper if inclusive else value >= upper)):
                raise ValueError(f"planner.dynamics.{name} is outside its supported range")
        if (not isinstance(dynamics.calibration_powers, tuple) or not dynamics.calibration_powers
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or not 0 < x <= 1
                       for x in dynamics.calibration_powers)):
            raise ValueError("planner.dynamics.calibration_powers requires powers in (0, 1]")
        # Posterior settings are dormant for frozen populations and analytical ST.
        if (dynamics.belief_update == "posterior" and dynamics.posterior_method != "standard") and (
                dynamics.validation_fraction <= 0 or dynamics.calibration_fraction <= 0
                or dynamics.calibration_target_coverage <= 0
                or dynamics.calibration_min_windows > dynamics.validation_max_windows):
            raise ValueError("calibrated posterior methods require separate calibration/test fractions and enough windows")
        for name, low, high in (("likelihood_power", 0, 1), ("validation_fraction", 0, 0.5)):
            value = getattr(dynamics, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not (low < value <= high if name == "likelihood_power" else low <= value < high)):
                raise ValueError(f"planner.dynamics.{name} is outside its supported range")
        for name, learner in (("risk", r), ("value", v)):
            if learner.input_mode not in ("state", "state_action"):
                raise ValueError(f"{name}.input_mode must be state or state_action")
        if not isinstance(r.enabled, bool):
            raise ValueError("risk.enabled must be true or false")
        if r.cutoff_mode not in ("learned", "manual"):
            raise ValueError("risk.cutoff_mode must be learned or manual")
        if (
            isinstance(r.manual_cutoff, bool)
            or not isinstance(r.manual_cutoff, (int, float))
            or not math.isfinite(r.manual_cutoff)
            or not 0 <= r.manual_cutoff <= 1
        ):
            raise ValueError("risk.manual_cutoff must be a finite probability in [0, 1]")
        if r.buffer_size is not None and (
            isinstance(r.buffer_size, bool)
            or not isinstance(r.buffer_size, int)
            or r.buffer_size < 1
        ):
            raise ValueError("risk.buffer_size must be null or a positive integer")
        for name, value, upper in (
            ("failure_fraction", r.failure_fraction, 1.0),
            ("validation_fraction", r.validation_fraction, 0.5),
            ("target_safe_fraction", r.target_safe_fraction, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"risk.{name} must be finite")
            if name == "validation_fraction":
                valid = 0 <= value < upper
            elif name == "target_safe_fraction":
                valid = 0 < value <= upper
            else:
                valid = 0 < value < upper
            if not valid:
                raise ValueError(f"risk.{name} is outside its supported range")
        if isinstance(r.batch_size, bool) or not isinstance(r.batch_size, int) or r.batch_size < 2:
            raise ValueError("risk.batch_size must be at least two for balanced batches")
        if not isinstance(p.model_bank, ModelBankConfig):
            raise ValueError("planner.model_bank must be a mapping with frictions")
        if not isinstance(p.risk_constraint, RiskConstraintConfig):
            raise ValueError("planner.risk_constraint must be a mapping")
        rc = p.risk_constraint
        if not isinstance(rc.enabled, bool) or rc.mode not in ("terminal", "all"):
            raise ValueError("planner.risk_constraint needs a boolean enabled and terminal/all mode")
        if rc.evaluation not in ("predictor", "distribution"):
            raise ValueError("planner.risk_constraint.evaluation must be predictor or distribution")
        if rc.enabled and not r.enabled:
            raise ValueError("planner.risk_constraint requires risk.enabled")
        for name in ("lambda_count", "max_candidates"):
            number = getattr(rc, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"planner.risk_constraint.{name} must be positive")
        if (
            isinstance(rc.lambda_max, bool)
            or not isinstance(rc.lambda_max, (int, float))
            or not math.isfinite(rc.lambda_max)
            or rc.lambda_max <= 0
        ):
            raise ValueError("planner.risk_constraint.lambda_max must be positive and finite")
        ps = p.probabilistic_safety
        if not isinstance(ps, ProbabilisticSafetyConfig) or not isinstance(ps.enabled, bool):
            raise ValueError("planner.probabilistic_safety needs a mapping with boolean enabled")
        if ps.fallback not in ("stop", "baseline"):
            raise ValueError("planner.probabilistic_safety.fallback must be stop or baseline")
        for name in ("boundary_threshold", "terminal_hull_threshold"):
            number = getattr(ps, name)
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or not 0 <= number <= 1):
                raise ValueError(f"planner.probabilistic_safety.{name} must be a probability in [0, 1]")
        for name in ("lambda_count", "max_candidates"):
            number = getattr(ps, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"planner.probabilistic_safety.{name} must be positive")
        if (isinstance(ps.lambda_max, bool) or not isinstance(ps.lambda_max, (int, float))
                or not math.isfinite(ps.lambda_max) or ps.lambda_max <= 0):
            raise ValueError("planner.probabilistic_safety.lambda_max must be positive and finite")
        frictions = p.model_bank.frictions
        if (
            not isinstance(frictions, (tuple, list))
            or not frictions
            or any(
                isinstance(mu, bool)
                or not isinstance(mu, (int, float))
                or not math.isfinite(mu)
                or mu <= 0
                for mu in frictions
            )
        ):
            raise ValueError(
                "planner.model_bank.frictions must be a nonempty list of positive finite numbers"
            )
        if len(set(frictions)) != len(frictions):
            raise ValueError("planner.model_bank.frictions must not contain duplicates")
        b = p.belief
        if not isinstance(b, BeliefConfig):
            raise ValueError("planner.belief must be a mapping")
        if not isinstance(b.enabled, bool) or b.backend not in ("model_bank", "ensemble_fit", "likelihood"):
            raise ValueError("planner.belief requires boolean enabled and backend=model_bank, ensemble_fit (fit_ensemble), or likelihood")
        if b.predictor not in ("fixed", "map", "ensemble"):
            raise ValueError("planner.belief.predictor must be fixed, map, or ensemble")
        if b.backend == "ensemble_fit" and b.predictor != "ensemble":
            raise ValueError("planner.belief ensemble_fit requires predictor=ensemble")
        if b.fit_horizon is not None and (
            isinstance(b.fit_horizon, bool) or not isinstance(b.fit_horizon, int) or b.fit_horizon < 1
        ):
            raise ValueError("planner.belief.fit_horizon must be null or a positive integer")
        for name in ("history_samples", "batch_size", "fit_max_iterations"):
            value = getattr(b, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"planner.belief.{name} must be a positive integer")
        for name in ("evidence_strength", "max_standardized_rmse", "fit_tolerance"):
            value = getattr(b, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"planner.belief.{name} must be positive and finite")
        if (isinstance(b.mean_fit_slack, bool) or not isinstance(b.mean_fit_slack, (int, float))
                or not math.isfinite(b.mean_fit_slack) or b.mean_fit_slack < 0):
            raise ValueError("planner.belief.mean_fit_slack must be nonnegative and finite")
        if isinstance(b.recent_fraction, bool) or not isinstance(b.recent_fraction, (int, float)) or not math.isfinite(b.recent_fraction) or not 0 <= b.recent_fraction <= 1:
            raise ValueError("planner.belief.recent_fraction must be in [0, 1]")
        if not isinstance(b.residual_scales, (tuple, list)) or len(b.residual_scales) != 7 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in b.residual_scales
        ):
            raise ValueError("planner.belief.residual_scales requires seven positive finite scales")
        if b.prior is not None and (
            not isinstance(b.prior, (tuple, list)) or len(b.prior) != len(frictions)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in b.prior)
            or sum(b.prior) <= 0
        ):
            raise ValueError("planner.belief.prior must match the bank with nonnegative finite positive-total weights")
        exp_name = self.output.exp_name
        if (
            not isinstance(exp_name, str)
            or not exp_name.strip()
            or exp_name != exp_name.strip()
            or exp_name in (".", "..")
            or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in exp_name)
        ):
            raise ValueError(
                "output.exp_name must be a nonempty folder label without path separators or reserved characters"
            )
        if s.model != "dynamic_ST":
            raise ValueError("Only dynamic_ST vehicle dynamics are supported")
        if not isinstance(p.sit_lmpc_mode, bool):
            raise ValueError("planner.sit_lmpc_mode must be true or false")
        if p.safe_set_selection not in ("exact", "predicted_window"):
            raise ValueError("planner.safe_set_selection must be exact or predicted_window")
        if p.safe_set_query not in ("facets", "implicit", "nf_density"):
            raise ValueError("planner.safe_set_query must be facets, implicit or nf_density")
        density = p.safe_set_density
        if not isinstance(density, SafeSetDensityConfig):
            raise ValueError("planner.safe_set_density must be a mapping")
        if density.method not in ("quantile", "sigma"):
            raise ValueError("planner.safe_set_density.method must be quantile or sigma")
        for name in ("hidden_dims", "layers", "bins", "epochs", "batch_size",
                     "min_training_states", "min_calibration_states", "max_states_per_partition",
                     "progress_bins"):
            if type(getattr(density, name)) is not int or getattr(density, name) < 1:
                raise ValueError(f"planner.safe_set_density.{name} must be a positive integer")
        if type(density.min_training_batches) is not int or density.min_training_batches < 0:
            raise ValueError("planner.safe_set_density.min_training_batches must be a nonnegative integer")
        if density.steps is not None and (type(density.steps) is not int or density.steps < 1):
            raise ValueError("planner.safe_set_density.steps must be a positive integer or null")
        if density.calibration not in ("global", "local"):
            raise ValueError("density calibration must be global or local")
        if density.penalty_normalization not in ("calibrated", "sample_max"):
            raise ValueError("density penalty_normalization must be calibrated or sample_max")
        if density.penalty_transform not in ("linear", "log1p"):
            raise ValueError("density penalty_transform must be linear or log1p")
        if (isinstance(density.penalty_upperbound_quantile, bool)
                or not isinstance(density.penalty_upperbound_quantile, (int, float))
                or not 0 < density.penalty_upperbound_quantile <= 1):
            raise ValueError("density penalty_upperbound_quantile must be in (0, 1]")
        if (isinstance(density.relaxation_quantile, bool)
                or not isinstance(density.relaxation_quantile, (int, float))
                or not 0 < density.relaxation_quantile <= 1):
            raise ValueError("density relaxation_quantile must be in (0, 1]")
        if (isinstance(density.sigma_cutoff_quantile, bool)
                or not isinstance(density.sigma_cutoff_quantile, (int, float))
                or not 0 < density.sigma_cutoff_quantile <= 1):
            raise ValueError("density sigma_cutoff_quantile must be in (0, 1]")
        if (isinstance(density.penalty_sigma, bool)
                or not isinstance(density.penalty_sigma, (int, float))
                or not math.isfinite(density.penalty_sigma) or density.penalty_sigma < 0):
            raise ValueError("density penalty_sigma must be finite and nonnegative")
        if (isinstance(density.learning_rate, bool) or not isinstance(density.learning_rate, (int, float))
                or not math.isfinite(density.learning_rate) or density.learning_rate <= 0):
            raise ValueError("density learning_rate must be positive and finite")
        for name in ("implicit_hull_point_bucket", "implicit_hull_max_iterations", "implicit_hull_block_size"):
            if type(getattr(p,name)) is not int or getattr(p,name)<1:
                raise ValueError(f"planner.{name} must be a positive integer")
        if (isinstance(p.implicit_hull_tolerance,bool) or not isinstance(p.implicit_hull_tolerance,(int,float))
                or not math.isfinite(p.implicit_hull_tolerance) or p.implicit_hull_tolerance<=0):
            raise ValueError("planner.implicit_hull_tolerance must be positive and finite")
        if (isinstance(p.safe_set_window_margin, bool)
                or not isinstance(p.safe_set_window_margin, (int, float))
                or not math.isfinite(p.safe_set_window_margin) or p.safe_set_window_margin < 0):
            raise ValueError("planner.safe_set_window_margin must be nonnegative and finite")
        if type(p.projection_block_size) is not int or p.projection_block_size < 1:
            raise ValueError("planner.projection_block_size must be a positive integer")
        if not isinstance(s.map, str) or not s.map.strip() or Path(s.map).name in ("", ".", ".."):
            raise ValueError("simulation.map must identify a map name or map prefix")
        if v.model not in ("normalizing_flow", "mlp"):
            raise ValueError("value.model must be normalizing_flow or mlp")
        if v.activation not in ("swish", "gelu", "tanh", "relu", "sin"):
            raise ValueError("unsupported value-model activation")
        if s.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if s.seed_mode not in ("auto", "fixed"):
            raise ValueError("seed_mode must be auto or fixed")
        if s.seed is not None and (
            isinstance(s.seed, bool) or not isinstance(s.seed, int) or not 0 <= s.seed < 2**32
        ):
            raise ValueError("seed must be null or an integer in [0, 2**32)")
        if s.seed_mode == "fixed" and s.seed is None:
            raise ValueError("fixed seed_mode requires a seed")
        for name in (
            "render",
            "plots",
            "summary_plots",
            "keyboard_controls",
            "keep_open_on_failure",
        ):
            if not isinstance(getattr(d, name), bool):
                raise ValueError(f"diagnostics.{name} must be true or false")
        for name in ("quick_load", "quick_load_init", "trust_legacy_init"):
            if not isinstance(getattr(self.output, name), bool):
                raise ValueError(f"output.{name} must be true or false")
        if self.output.quick_load and not self.output.resume and not self.output.init_path:
            raise ValueError("quick_load requires an explicit init_path")
        if (
            isinstance(self.output.checkpoint_every_trials, bool)
            or not isinstance(self.output.checkpoint_every_trials, int)
            or self.output.checkpoint_every_trials < 1
        ):
            raise ValueError("output.checkpoint_every_trials must be a positive integer")
        for name, number in {
            "max_laps": s.max_laps,
            "max_steps": s.max_steps,
            "horizon": p.horizon,
            "samples": p.samples,
            "iterations": p.iterations,
            "lambda_count": p.lambda_count,
            "safe_set_size": p.safe_set_size,
            "hull_bucket": p.hull_bucket,
            "hull_precompile_limit": p.hull_precompile_limit,
            "hull_selection_points": p.hull_selection_points,
            "safe_set_max_points": p.safe_set_max_points,
            "batch_size": v.batch_size,
            "epochs": v.epochs,
            "layers": v.layers,
            "hidden_dims": v.hidden_dims,
            "bins": v.bins,
            "value_samples": v.samples,
            "plot_interval": d.plot_interval,
            "risk.hidden_dims": r.hidden_dims,
            "risk.layers": r.layers,
            "risk.batch_size": r.batch_size,
            "risk.batches_per_epoch": r.batches_per_epoch,
            "risk.epochs": r.epochs,
            "risk.cutoff_min_states": r.cutoff_min_states,
            "risk.cutoff_min_episodes": r.cutoff_min_episodes,
        }.items():
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, number in {
            "trace_steps": d.trace_steps,
            "safe_set_min_states": p.safe_set_min_states,
            "boundary_samples": v.boundary_samples,
        }.items():
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, number in {
            "timestep": s.timestep,
            "integrator_timestep": s.integrator_timestep,
            "simulation.friction": s.friction,
            "simulation.mass_ratio": s.mass_ratio,
            "planner.friction": p.friction,
            "planner.mass_ratio": p.mass_ratio,
            "initial_speed": s.initial_speed,
            "temperature": p.temperature,
            "damping": p.damping,
            "learning_rate": v.learning_rate,
            "risk.learning_rate": r.learning_rate,
        }.items():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        ratio = s.timestep / s.integrator_timestep
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("timestep must be an integer multiple of integrator_timestep")
        if not 0 <= p.boundary_buffer < p.half_width:
            raise ValueError("boundary_buffer must be between zero and half_width")
        for name, number in {
            "half_width": p.half_width,
            "lambda_max": p.lambda_max,
            "safe_set_reward_limit": p.safe_set_reward_limit,
            "extension_seconds": p.extension_seconds,
            "reference_interval": p.reference_interval,
        }.items():
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if p.hull_selection_points < 6:
            raise ValueError(
                "hull_selection_points must be at least six for a five-dimensional hull"
            )
        if not isinstance(p.hull_precompile, bool):
            raise ValueError("hull_precompile must be true or false")
        if p.hull_precompile_limit <= p.hull_bucket:
            raise ValueError("hull_precompile_limit must exceed hull_bucket")
        if not math.isfinite(p.safe_set_relaxation) or not math.isfinite(p.action_rate_penalty):
            raise ValueError("planner penalties must be finite")
        for name, pair in {"action_scale": s.action_scale, "sample_std": p.sample_std}.items():
            if len(pair) != 2 or any(not math.isfinite(n) or n <= 0 for n in pair):
                raise ValueError(f"{name} must contain two positive finite values")
        if len(s.control_noise) != 2 or any(
            not math.isfinite(n) or n < 0 for n in (*s.control_noise, s.state_noise)
        ):
            raise ValueError("noise values must be nonnegative and finite")
        if p.lambda_max <= 0 or p.safe_set_relaxation < 0 or p.action_rate_penalty < 0:
            raise ValueError("invalid planner penalty or lambda range")
        return self

    def as_dict(self):
        return asdict(self)


SECTIONS = {
    "simulation": SimulationConfig,
    "planner": PlannerConfig,
    "value": ValueConfig,
    "diagnostics": DiagnosticsConfig,
    "output": OutputConfig,
    "risk": RiskConfig,
}


RETIRED_DIAGNOSTIC_FIELDS = ("keyboard_pause", "debug_plots", "model_comparison", "interactive")
RETIRED_DENSITY_FIELDS = ("local_min_calibration_states", "local_max_window_scale",
                          "min_safe_set_states", "cutoff_rule", "cutoff_sigma",
                          "cutoff_quantile", "target_safe_coverage", "relaxation_rule",
                          "relaxation_sigma", "penalty_rule",
                          "score_mean", "score_std")


def normalize_diagnostic_settings(settings):
    """One history limit; accept the old insight-only name when trace_steps is absent."""
    values = {key: value for key, value in settings.items()
              if key not in RETIRED_DIAGNOSTIC_FIELDS}
    if "insight_steps" in values:
        previous = values.pop("insight_steps")
        values.setdefault("trace_steps", previous)
    return values


def normalize_density_settings(settings, *, method=None):
    """Read retired fields without selecting a different NF method implicitly."""
    values = {key: value for key, value in settings.items() if key not in RETIRED_DENSITY_FIELDS}
    mode = values.get("method", method or "quantile")
    if mode == "sigma" and "cutoff_quantile" in settings:
        values.setdefault("sigma_cutoff_quantile", settings["cutoff_quantile"])
    elif mode != "sigma":
        # Old sigma settings were retired during the quantile-only period.
        # Reading such a file must not reactivate them without method=sigma.
        values.pop("penalty_sigma", None)
        values.pop("sigma_cutoff_quantile", None)
    if "penalty_quantile" in values:
        previous = values.pop("penalty_quantile")
        if "penalty_upperbound_quantile" in values and values["penalty_upperbound_quantile"] != previous:
            raise ValueError("conflicting penalty_quantile and penalty_upperbound_quantile values")
        values["penalty_upperbound_quantile"] = previous
    return values


def read_config_mapping(path=None, _seen=()):
    """Resolve a preset's optional relative `extends` before decoding its schema."""
    source = Path(str(files("sit_lmpc") / "config" / "default.yaml")) if path is None else Path(path).expanduser()
    source = source.resolve()
    if source in _seen:
        raise ValueError("configuration extends cycle: " + " -> ".join(map(str, (*_seen, source))))
    raw = yaml.safe_load(source.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("extends must be a nonempty YAML path")
    base = read_config_mapping(source.parent / parent, (*_seen, source))
    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
        return result
    return merge(base, raw)


def load_config(path=None, overrides=None):
    raw = read_config_mapping(path)
    if not isinstance(raw, dict) or set(raw) - (set(SECTIONS) | {"model_bank"}):
        raise ValueError(f"configuration sections must be {list(SECTIONS)}")
    # Normalize checkpoint-era top-level bank settings without editing saved files.
    if "model_bank" in raw:
        old_bank = raw.pop("model_bank")
        planner_values = raw.setdefault("planner", {})
        if not isinstance(planner_values, dict) or not isinstance(old_bank, dict):
            raise ValueError("model_bank and planner must be mappings")
        if "model_bank" in planner_values and planner_values["model_bank"] != old_bank:
            raise ValueError("conflicting model_bank and planner.model_bank values")
        planner_values["model_bank"] = old_bank
    # Old recorded runs placed the planner toggle under simulation. Read them
    # without retaining that location in newly resolved/saved configurations.
    if isinstance(raw.get("simulation"), dict) and "sit_lmpc_mode" in raw["simulation"]:
        old_mode = raw["simulation"].pop("sit_lmpc_mode")
        planner_values = raw.setdefault("planner", {})
        if not isinstance(planner_values, dict):
            raise ValueError("planner must be a mapping")
        if "sit_lmpc_mode" in planner_values and planner_values["sit_lmpc_mode"] != old_mode:
            raise ValueError("conflicting simulation/planner sit_lmpc_mode values")
        planner_values["sit_lmpc_mode"] = old_mode
    config = ExperimentConfig()
    for name, cls in SECTIONS.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
        if name == "planner" and "dynamics" in values:
            try:
                values["dynamics"] = DynamicsConfig(**values["dynamics"])
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Invalid planner.dynamics configuration: {error}") from error
        if name == "planner" and "safe_set_density" in values:
            try:
                # Read old trial/checkpoint configurations without carrying the
                # retired count/expansion policy into the sample-selected window.
                density_values = normalize_density_settings(values["safe_set_density"])
                values["safe_set_density"] = SafeSetDensityConfig(**density_values)
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Invalid planner.safe_set_density configuration: {error}") from error
        if name == "planner" and "model_bank" in values:
            bank = values["model_bank"]
            if not isinstance(bank, dict):
                raise ValueError("planner.model_bank must be a mapping")
            try:
                values["model_bank"] = ModelBankConfig(
                    **{
                        key: tuple(value) if isinstance(value, list) else value
                        for key, value in bank.items()
                    }
                )
            except TypeError as error:
                raise ValueError(f"Invalid planner.model_bank configuration: {error}") from error
        if name == "planner" and "risk_constraint" in values:
            try:
                values["risk_constraint"] = RiskConstraintConfig(**values["risk_constraint"])
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Invalid planner.risk_constraint configuration: {error}") from error
        if name == "planner" and "probabilistic_safety" in values:
            try:
                values["probabilistic_safety"] = ProbabilisticSafetyConfig(**values["probabilistic_safety"])
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Invalid planner.probabilistic_safety configuration: {error}") from error
        if name == "planner" and "belief" in values:
            try:
                values["belief"] = BeliefConfig(**{
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in values["belief"].items()
                })
            except (TypeError, AttributeError) as error:
                raise ValueError(f"Invalid planner.belief configuration: {error}") from error
        if name == "diagnostics":
            values = normalize_diagnostic_settings(values)
        # Old saved runs with only an integer seed retain their fixed-seed behavior.
        if name == "simulation" and values.get("seed") is not None and "seed_mode" not in values:
            values["seed_mode"] = "fixed"
        try:
            config = replace(config, **{name: cls(**values)})
        except TypeError as error:
            raise ValueError(f"Invalid {name} configuration: {error}") from error
    for name, values in (overrides or {}).items():
        if name not in SECTIONS:
            raise ValueError(f"unknown section {name}")
        if not isinstance(values, dict):
            raise ValueError(f"{name} overrides must be a mapping")
        # Resume passes YAML sequences through the override route as well.
        # Keep these fields immutable just as in the direct YAML-loading route.
        values = {key: tuple(value) if isinstance(value, list) else value
                  for key, value in values.items()}
        if name == "planner" and isinstance(values.get("dynamics"), dict):
            try:
                values = {**values, "dynamics": replace(config.planner.dynamics, **values["dynamics"])}
            except TypeError as error:
                raise ValueError(f"Invalid planner.dynamics override: {error}") from error
        if name == "planner" and isinstance(values.get("safe_set_density"), dict):
            try:
                values = {**values, "safe_set_density": replace(
                    config.planner.safe_set_density, **normalize_density_settings(
                        values["safe_set_density"], method=config.planner.safe_set_density.method))}
            except TypeError as error:
                raise ValueError(f"Invalid planner.safe_set_density override: {error}") from error
        if name == "diagnostics":
            values = normalize_diagnostic_settings(values)
        if name == "simulation" and values.get("seed") is not None and "seed_mode" not in values:
            values = {**values, "seed_mode": "fixed"}
        if name == "planner" and "model_bank" in values:
            bank = values["model_bank"]
            if isinstance(bank, dict):
                try:
                    bank = ModelBankConfig(**{
                        key: tuple(value) if isinstance(value, list) else value
                        for key, value in bank.items()
                    })
                except TypeError as error:
                    raise ValueError(f"Invalid planner.model_bank override: {error}") from error
            values = {**values, "model_bank": bank}
        if name == "planner" and "risk_constraint" in values:
            constraint = values["risk_constraint"]
            if isinstance(constraint, dict):
                try:
                    constraint = replace(config.planner.risk_constraint, **constraint)
                except TypeError as error:
                    raise ValueError(f"Invalid planner.risk_constraint override: {error}") from error
            values = {**values, "risk_constraint": constraint}
        if name == "planner" and "probabilistic_safety" in values:
            safety = values["probabilistic_safety"]
            if isinstance(safety, dict):
                try:
                    safety = replace(config.planner.probabilistic_safety, **safety)
                except TypeError as error:
                    raise ValueError(f"Invalid planner.probabilistic_safety override: {error}") from error
            values = {**values, "probabilistic_safety": safety}
        if name == "planner" and "belief" in values:
            belief = values["belief"]
            if isinstance(belief, dict):
                try:
                    belief = replace(getattr(config, name).belief, **{
                        key: tuple(value) if isinstance(value, list) else value
                        for key, value in belief.items()
                    })
                except TypeError as error:
                    raise ValueError(f"Invalid planner.belief override: {error}") from error
            values = {**values, "belief": belief}
        config = replace(config, **{name: replace(getattr(config, name), **values)})
    return config.validate()
