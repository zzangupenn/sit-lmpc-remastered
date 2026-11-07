"""Frozen configuration and explicit, reproducible defaults."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import secrets
import yaml


def _range(name, pair, positive=False):
    if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or pair[0] > pair[1]:
        raise ValueError(f"{name} must be a finite ordered pair")
    if positive and pair[0] <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CollectionConfig:
    source: str = "gym_st"
    profile: str = "fullscale"
    friction: tuple[float, float] = (0.3, 1.2)
    mass_ratio: tuple[float, float] = (0.8, 1.2)
    friction_count: int = 5
    mass_count: int = 3
    random_instances: int = 6
    inertia_policy: str = "fixed"
    trajectories: int = 6
    steps: int = 80
    timestep: float = 0.1
    integrator_timestep: float = 0.02
    speed: tuple[float, float] = (2.0, 15.0)
    low_speed_fraction: float = 0.1
    low_speed_range: tuple[float, float] = (0.1, 0.4)
    steering: tuple[float, float] = (-0.12, 0.12)
    yaw_rate: tuple[float, float] = (-0.12, 0.12)
    slip: tuple[float, float] = (-0.03, 0.03)
    control_scale: tuple[float, float] = (0.4, 3.5)
    actuator_noise: tuple[float, float] = (0.0, 0.0)
    measurement_noise: tuple[float, ...] = (0.0,) * 7
    max_abs_slip: float = 0.8
    max_abs_yaw_rate: float = 4.0
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    extrapolation_instances: int = 2
    map: str = ""
    seed: int | None = None
    output: str = "datasets/st"

    def validate(self):
        if self.source not in {"gym_st", "gym_env"}:
            raise ValueError("source must be gym_st or gym_env")
        if self.profile not in {"fullscale", "f1tenth"}:
            raise ValueError("profile must be fullscale or f1tenth")
        if self.inertia_policy not in {"fixed", "proportional"}:
            raise ValueError("inertia_policy must be fixed or proportional")
        for name in ("friction", "mass_ratio", "speed"):
            _range(name, getattr(self, name), positive=True)
        for name in ("steering", "yaw_rate", "slip"):
            _range(name, getattr(self, name))
        if not 0 <= self.low_speed_fraction <= 1:
            raise ValueError("low_speed_fraction must be in [0, 1]")
        _range("low_speed_range", self.low_speed_range, positive=True)
        if self.low_speed_range[1] >= 0.5:
            raise ValueError("low_speed_range must remain below Gym's 0.5 m/s switch")
        for name in ("friction_count", "mass_count", "trajectories", "steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("random_instances", "extrapolation_instances"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not (
            0 < self.train_fraction < 1
            and 0 < self.validation_fraction < 1
            and self.train_fraction + self.validation_fraction < 1
        ):
            raise ValueError("train/validation fractions must leave a nonempty test fraction")
        _interval(self.timestep, self.integrator_timestep)
        _noise(self.actuator_noise, 2, "actuator_noise")
        _noise(self.measurement_noise, 7, "measurement_noise")
        if len(self.control_scale) != 2 or not all(
            math.isfinite(x) and x > 0 for x in self.control_scale
        ):
            raise ValueError("control_scale must contain two positive values")
        if not all(math.isfinite(x) and x > 0 for x in (self.max_abs_slip, self.max_abs_yaw_rate)):
            raise ValueError("collection guardrails must be positive")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class VehicleCollectionConfig:
    """Paired ST study using the smooth-target/PID vehicle-data excitation."""

    profile: str = "fullscale"
    friction: tuple[float, float] = (0.3, 1.0)
    front_cornering_scale: tuple[float, float] = (0.9, 1.1)
    rear_cornering_scale: tuple[float, float] = (0.9, 1.1)
    mass_ratio: tuple[float, float] = (0.9, 1.1)
    random_instances: int = 24
    speeds: tuple[float, ...] = tuple(float(value) for value in range(5, 21))
    signals_per_speed: int = 2
    steps: int = 120
    transitions_per_instance: int | None = None
    timestep: float = 0.1
    integrator_timestep: float = 0.01
    steering_sines: int = 10
    lateral_accel_target: float = 2.0
    speed_target_span: float = 3.0
    speed_target_refresh: int = 42
    controller_noise_scale: float = 0.25
    max_abs_slip: float = 0.8
    max_abs_yaw_rate: float = 4.0
    seed: int | None = 20261016
    output: str = "runs/vehicle_st/paired-study"

    def validate(self):
        if self.profile != "fullscale":
            raise ValueError("vehicle study currently uses the fullscale profile")
        for name in ("friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio"):
            _range(name, getattr(self, name), positive=True)
        if not self.speeds or any(not math.isfinite(x) or not 0 < x <= 20 for x in self.speeds):
            raise ValueError("speeds must be positive and at most 20 m/s")
        for name in (
            "random_instances", "signals_per_speed", "steps", "steering_sines",
            "speed_target_refresh",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.steering_sines < 2:
            raise ValueError("steering_sines must be at least 2")
        if self.transitions_per_instance is not None:
            target = self.transitions_per_instance
            if type(target) is not int or target < 1 or target % len(self.speeds):
                raise ValueError("transitions_per_instance must divide evenly across speeds")
            final_segment = target // len(self.speeds) - (self.signals_per_speed - 1) * self.steps
            if not 1 <= final_segment <= self.steps:
                raise ValueError("transitions_per_instance does not fit the configured segments")
        _interval(self.timestep, self.integrator_timestep)
        for name in (
            "lateral_accel_target", "speed_target_span", "controller_noise_scale",
            "max_abs_slip", "max_abs_yaw_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lateral_accel_target == 0 or self.max_abs_slip == 0 or self.max_abs_yaw_rate == 0:
            raise ValueError("lateral acceleration and trajectory guardrails must be positive")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class VehicleStudyConfig:
    """Add unseen ST parameter combinations to an existing paired vehicle dataset."""

    source_dataset: str = "runs/vehicle_st/paired-study-50k"
    output: str = "runs/vehicle_st/bnn-study-50k"
    unseen_instances: int = 16
    validation_instances: int = 6
    seed: int | None = 20261017

    def validate(self):
        for name in ("unseen_instances", "validation_instances"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if Path(self.source_dataset).resolve() == Path(self.output).resolve():
            raise ValueError("study output must differ from the source dataset")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class VehiclePopulationStudyConfig:
    """Generate new train/validation physics while preserving an existing test set."""

    source_dataset: str = "runs/vehicle_st/bnn-study-50k"
    output: str = "runs/vehicle_st/bnn-study-100-10-16"
    training_instances: int = 100
    validation_instances: int = 10
    seed: int | None = 20261020

    def validate(self):
        for name in ("training_instances", "validation_instances"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if Path(self.source_dataset).resolve() == Path(self.output).resolve():
            raise ValueError("study output must differ from the source dataset")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class ModelConfig:
    domain: str = "st"
    bayesian: bool = True
    mode: str = "dynamics"
    latent_dimension: int = 8
    hidden_sizes: tuple[int, ...] = (64, 64)
    dtype: str = "float32"
    learn_prior: bool = True
    learn_noise: bool = True
    noise_std: float = 0.05
    noise_floor: float = 0.003
    noise_ceiling: float = 0.5
    prior_floor: float = 0.001
    timestep: float = 0.1
    integrator_timestep: float = 0.02
    baseline_profile: str = "fullscale"
    baseline_friction: float = 0.75
    baseline_mass_ratio: float = 1.0
    baseline_inertia_policy: str = "fixed"

    def validate(self):
        from dyna_dist.domains import get_domain

        get_domain(self.domain)
        if self.domain != "st" and self.mode != "dynamics":
            raise ValueError(
                "Non-ST domains currently support full dynamics only, not an ST residual baseline"
            )
        if self.mode not in {"dynamics", "residual"}:
            raise ValueError("mode must be dynamics or residual")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not self.hidden_sizes or not all(
            type(x) is int and x > 0 for x in (self.latent_dimension, *self.hidden_sizes)
        ):
            raise ValueError("network dimensions must be positive")
        if not 0 < self.noise_floor < self.noise_std < self.noise_ceiling:
            raise ValueError("require 0 < noise_floor < noise_std < noise_ceiling")
        if not 0 < self.prior_floor < 1:
            raise ValueError("prior_floor must be in (0, 1)")
        if self.baseline_profile not in {"fullscale", "f1tenth"}:
            raise ValueError("invalid baseline_profile")
        if self.baseline_inertia_policy not in {"fixed", "proportional"}:
            raise ValueError("invalid baseline_inertia_policy")
        if not all(
            math.isfinite(x) and x > 0 for x in (self.baseline_friction, self.baseline_mass_ratio)
        ):
            raise ValueError("baseline friction/mass must be positive")
        _interval(self.timestep, self.integrator_timestep)
        return self


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str = "datasets/st"
    output: str = "runs/train"
    seed: int | None = None
    steps: int = 2000
    batch_instances: int = 8
    objective: str = "context_predictive"
    training_option: int = 1
    prior_refit: bool = False
    refit_transitions: int = 64
    refit_iterations: int = 1
    em_refresh_every: int = 200
    em_transitions: int = 0
    em_chunk_size: int = 4096
    rollout_validation_every: int = 1000
    rollout_validation_particles: int = 64
    context_min: int = 0
    context_max: int = 32
    query_size: int = 32
    learning_rate: float = 0.001
    gradient_clip: float = 5.0
    weight_decay: float = 0.00001
    validation_every: int = 100
    checkpoint_every: int = 100
    validation_batches: int = 8
    progress_plot: bool = False
    data_view: str = "true"
    max_transitions_per_instance: int = 0
    balance_learning_stages: bool = False
    resume_path: str | None = None
    init_model_path: str | None = None
    model: ModelConfig = ModelConfig()

    def validate(self):
        self.model.validate()
        if self.training_option not in (1, 2) or type(self.training_option) is not int:
            raise ValueError("training_option must be 1 or 2")
        if type(self.prior_refit) is not bool:
            raise ValueError("prior_refit must be a boolean")
        if self.training_option == 2 or self.prior_refit:
            if self.objective != "population_joint" or not self.model.bayesian:
                raise ValueError("option 2 and prior refit require Bayesian population_joint")
            if not self.model.learn_prior:
                raise ValueError("option 2 and prior refit require learn_prior=true")
        if self.training_option == 2 and self.prior_refit:
            raise ValueError("option 2 updates the population during training; disable prior_refit")
        if self.objective not in {"context_predictive", "population_joint"}:
            raise ValueError("objective must be context_predictive or population_joint")
        for name in (
            "steps",
            "batch_instances",
            "query_size",
            "validation_every",
            "checkpoint_every",
            "validation_batches",
            "refit_transitions", "refit_iterations", "em_refresh_every", "em_chunk_size",
            "rollout_validation_every", "rollout_validation_particles",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.em_transitions) is not int or self.em_transitions < 0:
            raise ValueError("em_transitions must be nonnegative; zero uses all train transitions")
        if not all(type(x) is int for x in (self.context_min, self.context_max)) or not (
            0 <= self.context_min <= self.context_max
        ):
            raise ValueError("require 0 <= context_min <= context_max")
        if self.objective == "population_joint" and (self.context_min or self.context_max):
            raise ValueError("population_joint requires context_min=context_max=0")
        if self.resume_path and self.init_model_path:
            raise ValueError("resume_path and init_model_path are mutually exclusive")
        if not all(
            math.isfinite(x) for x in (self.learning_rate, self.gradient_clip, self.weight_decay)
        ) or (self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0):
            raise ValueError("optimizer settings must be positive/nonnegative")
        if self.data_view not in {"true", "measured"}:
            raise ValueError("data_view must be true or measured")
        if type(self.progress_plot) is not bool:
            raise ValueError("progress_plot must be a boolean")
        if (
            type(self.max_transitions_per_instance) is not int
            or self.max_transitions_per_instance < 0
        ):
            raise ValueError("max_transitions_per_instance must be a nonnegative integer")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: str = "datasets/st"
    model_path: str = "runs/train/best"
    output: str = "runs/evaluation"
    split: str = "test"
    conditioning_mode: str = "context"
    context_sizes: tuple[int, ...] = (0, 4, 16, 32)
    particles: int = 64
    horizon: int = 10
    plots: bool = False
    seed: int | None = None

    def validate(self):
        if self.split not in {"train", "validation", "test", "extrapolation"}:
            raise ValueError("invalid evaluation split")
        if self.conditioning_mode not in {"context", "prior"}:
            raise ValueError("conditioning_mode must be context or prior")
        if not self.context_sizes or not all(type(x) is int and x >= 0 for x in self.context_sizes):
            raise ValueError("context_sizes must be nonnegative")
        if self.conditioning_mode == "prior" and self.context_sizes != (0,):
            raise ValueError("prior evaluation requires context_sizes=(0,)")
        if not all(type(x) is int and x > 0 for x in (self.particles, self.horizon)):
            raise ValueError("particle count/horizon must be positive")
        _seed(self.seed)
        return self


@dataclass(frozen=True)
class VehicleBNNEvaluationConfig:
    """Prior-only comparison against fresh physical ST function samples."""

    dataset: str = "runs/vehicle_st/bnn-study-50k"
    training_run: str = "runs/vehicle_st/bnn-training-stable-20261016-0054"
    output: str = "runs/vehicle_st/bnn-evaluation"
    latent_dimensions: tuple[int, ...] = (4, 8, 16, 32)
    seed: int | None = 20261019
    one_step_per_speed: int = 128
    joint_groups: int = 4
    joint_group_size: int = 64
    rollout_speeds: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
    rollout_signals: tuple[int, ...] = (0, 1, 2)
    horizon: int = 15
    report_horizons: tuple[int, ...] = (1, 5, 10, 15)
    rollout_starts: tuple[int, ...] = (0, 40)
    model_particles: int = 256
    real_models: int = 256
    mppi_enabled: bool = True
    mppi_candidates: int = 1024
    mppi_particles: int = 64
    mppi_sample_std: tuple[float, float] = (0.25, 0.25)
    mppi_action_scale: tuple[float, float] = (0.4, 3.5)
    mppi_half_widths: tuple[float, ...] = (0.5, 1.0, 2.0)
    mppi_probability_threshold: float = 0.05
    mppi_temperature: float = 0.01
    benchmark_particles: tuple[int, ...] = (32, 64, 128)
    replan_steps: int = 15
    replan_candidates: int = 256

    def validate(self):
        for name in (
            "one_step_per_speed", "joint_groups", "joint_group_size", "horizon",
            "model_particles", "real_models", "mppi_candidates", "mppi_particles",
            "replan_steps", "replan_candidates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.latent_dimensions or any(type(x) is not int or x < 1 for x in self.latent_dimensions):
            raise ValueError("latent_dimensions must be positive integers")
        if len(set(self.latent_dimensions)) != len(self.latent_dimensions):
            raise ValueError("latent_dimensions must be distinct")
        if not self.rollout_speeds or any(not math.isfinite(x) or x <= 0 for x in self.rollout_speeds):
            raise ValueError("rollout_speeds must be finite positive speeds")
        if not self.rollout_signals or any(type(x) is not int or x < 0 for x in self.rollout_signals):
            raise ValueError("rollout_signals must be nonnegative integers")
        if not self.report_horizons or any(
            type(x) is not int or x < 1 or x > self.horizon for x in self.report_horizons
        ):
            raise ValueError("report_horizons must be within the rollout horizon")
        if len(set(self.report_horizons)) != len(self.report_horizons):
            raise ValueError("report_horizons must be distinct")
        if self.horizon not in self.report_horizons:
            raise ValueError("report_horizons must include the maximum horizon")
        if not self.rollout_starts or any(type(x) is not int or x < 0 for x in self.rollout_starts):
            raise ValueError("rollout_starts must be nonnegative integers")
        if not set(self.benchmark_particles) or any(
            type(x) is not int or x < 1 for x in self.benchmark_particles
        ):
            raise ValueError("benchmark_particles must be positive integers")
        for name in ("mppi_sample_std", "mppi_action_scale"):
            values = getattr(self, name)
            if len(values) != 2 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"{name} must contain two positive finite values")
        if not self.mppi_half_widths or any(
            not math.isfinite(x) or x <= 0 for x in self.mppi_half_widths
        ):
            raise ValueError("mppi_half_widths must be positive finite values")
        if not 0 <= self.mppi_probability_threshold <= 1:
            raise ValueError("mppi_probability_threshold must be within [0, 1]")
        if not math.isfinite(self.mppi_temperature) or self.mppi_temperature <= 0:
            raise ValueError("mppi_temperature must be positive and finite")
        _seed(self.seed)
        return self


def _interval(dt, substep):
    if not all(math.isfinite(x) and x > 0 for x in (dt, substep)):
        raise ValueError("timesteps must be positive and finite")
    if not math.isclose(dt / substep, round(dt / substep)) or substep > dt:
        raise ValueError("timestep must be an integer multiple of integrator_timestep")


def _noise(values, count, name):
    if len(values) != count or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{name} must contain {count} finite nonnegative values")


def _seed(seed):
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be null or an integer in [0, 2**32)")


def resolve_seed(seed):
    _seed(seed)
    return secrets.randbits(32) if seed is None else seed


def decode(cls, raw):
    raw = dict(raw)
    allowed = {f.name for f in fields(cls)}
    if set(raw) - allowed:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(set(raw) - allowed)}")
    if cls is TrainingConfig and "model" in raw:
        raw["model"] = decode(ModelConfig, raw["model"])
    for field in fields(cls):
        if field.name in raw and isinstance(field.default, tuple):
            raw[field.name] = tuple(raw[field.name])
    return cls(**raw).validate()


def load_config(cls, path=None, overrides=None):
    raw = {} if path is None else yaml.safe_load(Path(path).read_text()) or {}
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return decode(cls, raw)


def write_config(config, path):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
