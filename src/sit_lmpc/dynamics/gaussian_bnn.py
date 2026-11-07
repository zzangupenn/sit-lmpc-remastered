"""Coherent dyna-dist functions, optionally conditioned by the online learner."""

from dataclasses import dataclass, replace
from functools import cached_property, partial
from pathlib import Path
import math
import shutil

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from dyna_dist.inference.posterior import FunctionSamples
from dyna_dist.io.checkpoint import load_model
from dyna_dist.models.features import prior_moments_and_variance
from .distribution import TrajectoryScenarios


@partial(jax.jit, static_argnames=("count",))
def sample_coefficients(mean, factor, key, count):
    # Preserve the former normal -> matmul -> addition rounding boundaries.
    # Otherwise fusion changes many coefficient bits in concentrated posteriors.
    normal = jax.lax.optimization_barrier(
        jax.random.normal(key, (count, len(mean)), dtype=mean.dtype))
    correction = jax.lax.optimization_barrier(
        jnp.matmul(normal, factor.T, precision=jax.lax.Precision.HIGHEST))
    return mean + correction


@partial(jax.jit, static_argnames=("count",))
def sample_prior_coefficients(model, key, count):
    mean, factor, _ = prior_moments_and_variance(model.parameters, model.config)
    return sample_coefficients(mean, factor, key, count)


@struct.dataclass
class GaussianBNN:
    functions: FunctionSamples
    action_scale: jax.Array
    posterior_mean: jax.Array | None = None
    posterior_factor: jax.Array | None = None
    belief_generation: jax.Array = 0
    likelihood_power: jax.Array = 1.0
    posterior_method: str = struct.field(pytree_node=False, default="standard")

    @property
    def params(self):
        # Common planner dtype handle, not physical ST parameters.
        return self.functions.coefficients

    @property
    def weights(self):
        return self.functions.weights

    @property
    def timestep(self):
        return self.functions.model.config.timestep

    @property
    def integrator_timestep(self):
        return self.functions.model.config.integrator_timestep

    def resample(self, key, count=None):
        count = len(self.weights) if count is None else count
        if self.posterior_mean is not None:
            coefficients = sample_coefficients(self.posterior_mean, self.posterior_factor, key, count)
        else:
            coefficients = sample_prior_coefficients(self.functions.model, key, count)
        # Only coefficients are outputs of the sampler kernel. Preserve snapshot
        # array identities/ownership rather than returning the whole model via JIT.
        return self.replace(functions=FunctionSamples(self.functions.model, coefficients))

    def rollout_members(self, state, actions):
        return self.functions.rollout(state, actions * self.action_scale)[0]

    def step_members(self, state, action):
        return self.functions.step(state, action * self.action_scale)

    def context(self):
        model = self.functions.model
        return {
            "kind": "gaussian_bnn", "particles": len(self.weights),
            "latent_dimension": model.config.latent_dimension,
            "model_fingerprint": model.fingerprint, "population_generation": model.generation,
            "target_mode": model.config.mode,
            "adaptation": "frozen_population" if self.posterior_mean is None else "gaussian_posterior",
            "belief_generation": int(self.belief_generation),
            "likelihood_power": float(self.likelihood_power),
            "posterior_method": self.posterior_method,
            "inference": ("calibrated_working_likelihood" if self.posterior_method != "standard"
                          else "conditional_bayes" if float(self.likelihood_power) == 1 else "generalized_bayes"),
        }


def load_predictor(config, simulation, key, *, source=None, destination=None):
    if source is None and config.checkpoint is None:
        raise ValueError("gaussian_bnn needs a checkpoint path or bundled resume artifact")
    source = Path(source or config.checkpoint).expanduser().resolve()
    population = load_model(source)
    if population.config.domain != "st" or not population.config.bayesian:
        raise ValueError("SIT-LMPC Remastered requires a Bayesian ST-domain dyna-dist model")
    if not math.isclose(population.config.timestep, simulation.timestep, rel_tol=1e-9):
        raise ValueError("BNN transition interval differs from simulation.timestep")
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(exist_ok=True)
        # Keep the trained artifact unchanged; a runtime precision cast is not training.
        for name in ("manifest.json", "model.msgpack"):
            if source / name != destination.resolve() / name:
                shutil.copy2(source / name, destination / name)
    dtype = getattr(jnp, simulation.dtype)
    population = jax.tree.map(
        lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        population,
    )
    population = population.replace(config=replace(population.config, dtype=simulation.dtype))
    return GaussianBNN(population.sample_functions(key, config.particles),
                       jnp.asarray(simulation.action_scale, dtype=dtype))


@dataclass(frozen=True)
class BNNScenarioProvider:
    model: GaussianBNN
    verification_particles: int = 2048
    verification_confidence: float = 0.95
    chance_estimation: str = "independent"

    def __post_init__(self):
        if self.chance_estimation not in ("independent", "planning_samples"):
            raise ValueError("chance_estimation must be independent or planning_samples")

    @property
    def uses_independent_verification(self):
        return self.chance_estimation == "independent"

    @property
    def scenario_count(self):
        return len(self.model.weights)

    @cached_property
    def metadata(self):
        independent = self.uses_independent_verification
        return {
            **self.model.context(), "method": "gaussian_last_layer_mc",
            "sample_count": self.scenario_count,
            "probability_source": "gaussian_population_prior" if self.model.posterior_mean is None else "gaussian_latent_posterior",
            "belief_generation": int(self.model.belief_generation),
            "chance_estimation": self.chance_estimation,
            "probability_estimator": "independent_mc_upper" if independent else "empirical_planning_samples",
            "verification_sample_count": self.verification_particles if independent else 0,
            "verification_confidence": self.verification_confidence if independent else None,
            "confidence": self.verification_confidence if independent else None,
            "uncertainty": "learned population" if self.model.posterior_mean is None else "conditional Gaussian belief; fixed learned features/likelihood",
            "limitations": ("execution noise excluded; not real-system certification" if independent else
                            "optimization samples reused; no independent confidence bound; execution noise excluded; not real-system certification"),
        }

    def with_model(self, model):
        provider = replace(self, model=model)
        previous, current = self.model, model
        if (previous.functions.model is current.functions.model
                and previous.posterior_mean is current.posterior_mean
                and previous.posterior_factor is current.posterior_factor
                and previous.belief_generation is current.belief_generation
                and previous.likelihood_power is current.likelihood_power
                and len(previous.weights) == len(current.weights)
                and "metadata" in self.__dict__):
            provider.__dict__["metadata"] = self.__dict__["metadata"]
        return provider

    def can_reuse(self, model):
        return isinstance(model, GaussianBNN)

    def from_members(self, members):
        finite = jnp.all(jnp.isfinite(members), axis=-1)
        return TrajectoryScenarios(members, self.model.weights,
                                   jnp.cumprod(finite, axis=-1).astype(bool), None)

    def rollout(self, state, controls, *, key=None, scan=True):
        del key, scan  # The planner already froze one seeded function batch.
        members = self.model.functions.rollout(state, controls * self.model.action_scale)
        return self.from_members(members)

    def verification_rollout(self, state, controls, *, planning_key, validation_key):
        if not self.uses_independent_verification:
            raise ValueError("planning_samples reuses planning paths; independent verification is disabled")
        if np.array_equal(jax.random.key_data(planning_key), jax.random.key_data(validation_key)):
            raise ValueError("verification must use an independent reserved key")
        model = self.model.resample(validation_key, self.verification_particles)
        members = model.functions.rollout(state, controls * model.action_scale)
        finite = jnp.all(jnp.isfinite(members), axis=-1)
        # Two one-sided geometric events across this fixed, pre-validation candidate set.
        error = min(1.0, math.sqrt(
            math.log(2 * len(controls) / (1 - self.verification_confidence))
            / (2 * self.verification_particles)
        ))
        return TrajectoryScenarios(
            members, model.weights, jnp.cumprod(finite, axis=-1).astype(bool),
            jnp.asarray(error, dtype=members.dtype),
        )
