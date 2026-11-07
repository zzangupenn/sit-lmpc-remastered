"""Frozen population model, Gaussian posterior, and coherent batched dynamics."""

from dataclasses import dataclass
import math
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from dyna_dist.data.schema import Transitions
from dyna_dist.data.transforms import features, increments, reconstruct, global_covariance
from dyna_dist.models.features import network, prior_and_variance, prior_moments_and_variance
from dyna_dist.models.gaussian import condition as gaussian_condition
from dyna_dist.models.st import step as st_step
from dyna_dist.domains import get_domain


@struct.dataclass
class PopulationModel:
    parameters: object
    normalizer: object
    baseline_parameters: jax.Array
    config: object = struct.field(pytree_node=False)
    generation: int = struct.field(pytree_node=False, default=0)
    fingerprint: str = struct.field(pytree_node=False, default="unsaved")

    def prior(self):
        state, _ = prior_and_variance(self.parameters, self.config)
        return Posterior(self, state)

    def sample_functions(self, key, count):
        """Draw coherent dynamics functions directly from the learned population."""
        if type(count) is not int or count < 1:
            raise ValueError("sample count must be a positive integer")
        mean, factor, _ = prior_moments_and_variance(self.parameters, self.config)
        coefficients = mean + jnp.matmul(
            jax.random.normal(key, (count, len(mean)), dtype=mean.dtype), factor.T,
            precision=jax.lax.Precision.HIGHEST,
        )
        return FunctionSamples(self, coefficients)

    @property
    def domain(self):
        return get_domain(self.config.domain)

    def targets(self, states, controls, next_states):
        return self.normalizer.normalize_targets(
            increments(states, next_states, self.config.domain)
            - baseline_increments(self, states, controls)
        )


@jax.jit
def baseline_increments(model, states, controls):
    if model.config.mode == "dynamics":
        return jnp.zeros_like(states)
    shape = states.shape
    targets = jax.vmap(
        lambda x, u: st_step(
            x, u, model.baseline_parameters, model.config.timestep, model.config.integrator_timestep
        )
    )(states.reshape(-1, 7), controls.reshape(-1, 2))
    return increments(states, targets.reshape(shape))


@jax.jit
def components(model, states, controls):
    mean, basis = network(model.config).apply(
        {"params": model.parameters["network"]},
        model.normalizer.normalize_inputs(features(states, controls, model.config.domain)),
    )
    return mean, basis if model.config.bayesian else jnp.zeros_like(basis)


@jax.jit
def _predict(model, state, states, controls):
    b, basis = components(model, states, controls)
    mean, covariance = state.moments()
    normalized = b + jnp.einsum("...kd,d->...k", basis, mean, precision=jax.lax.Precision.HIGHEST)
    delta = model.normalizer.denormalize_targets(normalized) + baseline_increments(
        model, states, controls
    )
    physical_basis = basis * model.normalizer.target_scale[..., None]
    epistemic = jnp.einsum(
        "...kd,de,...le->...kl",
        physical_basis,
        covariance,
        physical_basis,
        precision=jax.lax.Precision.HIGHEST,
    )
    _, variance = prior_and_variance(model.parameters, model.config)
    observation = jnp.diag(variance * model.normalizer.target_scale**2)
    observation = jnp.broadcast_to(observation, epistemic.shape)
    return Prediction(
        reconstruct(states, delta, model.config.domain),
        global_covariance(states, epistemic, model.config.domain),
        global_covariance(states, observation, model.config.domain),
    )


@jax.jit
def _predict_prior(model, states, controls):
    """Use the stored Gaussian factor, avoiding an ill-conditioned precision round trip."""
    b, basis = components(model, states, controls)
    mean, factor, variance = prior_moments_and_variance(model.parameters, model.config)
    normalized = b + jnp.einsum("...kd,d->...k", basis, mean,
                                 precision=jax.lax.Precision.HIGHEST)
    delta = model.normalizer.denormalize_targets(normalized) + baseline_increments(model, states, controls)
    weighted = jnp.matmul(basis * model.normalizer.target_scale[..., None], factor,
                          precision=jax.lax.Precision.HIGHEST)
    epistemic = jnp.matmul(weighted, jnp.swapaxes(weighted, -1, -2),
                            precision=jax.lax.Precision.HIGHEST)
    observation = jnp.broadcast_to(jnp.diag(variance * model.normalizer.target_scale**2), epistemic.shape)
    return Prediction(reconstruct(states, delta, model.config.domain),
                      global_covariance(states, epistemic, model.config.domain),
                      global_covariance(states, observation, model.config.domain))


@struct.dataclass
class Prediction:
    mean: jax.Array
    epistemic_covariance: jax.Array
    observation_covariance: jax.Array

    @property
    def predictive_covariance(self):
        return self.epistemic_covariance + self.observation_covariance


@jax.jit
def _function_step(model, state, control, coefficients):
    b, basis = components(model, state, control)
    normalized = b + jnp.einsum("kd,d->k", basis, coefficients, precision=jax.lax.Precision.HIGHEST)
    delta = model.normalizer.denormalize_targets(normalized) + baseline_increments(
        model, state, control
    )
    return reconstruct(state, delta, model.config.domain)


@jax.jit
def _particle_steps(model, coefficients, states, controls):
    return jax.vmap(lambda x, u, z: _function_step(model, x, u, z))(states, controls, coefficients)


@jax.jit
def _rollouts(model, coefficients, initial_states, controls):
    def candidate(initial, actions):
        initial = jnp.broadcast_to(initial, (len(coefficients), model.domain.state_dimension))

        def advance(states, action):
            target = _particle_steps(
                model,
                coefficients,
                states,
                jnp.broadcast_to(action, (len(coefficients), model.domain.control_dimension)),
            )
            return target, target

        return jax.lax.scan(advance, initial, actions)[1].swapaxes(0, 1)

    return jax.vmap(candidate)(initial_states, controls)


@struct.dataclass
class FunctionSamples:
    model: PopulationModel
    coefficients: jax.Array

    @property
    def weights(self):
        return jnp.full(
            len(self.coefficients), 1 / len(self.coefficients), dtype=self.coefficients.dtype
        )

    def step(self, state, physical_control):
        dtype = self.coefficients.dtype
        count = len(self.coefficients)
        states = jnp.broadcast_to(
            jnp.asarray(state, dtype=dtype), (count, self.model.domain.state_dimension)
        )
        controls = jnp.broadcast_to(
            jnp.asarray(physical_control, dtype=dtype), (count, self.model.domain.control_dimension)
        )
        return _particle_steps(self.model, self.coefficients, states, controls)

    def rollout(self, initial_state, controls):
        controls = jnp.asarray(controls, dtype=self.coefficients.dtype)
        if controls.ndim == 2:
            controls = controls[None]
        if (
            controls.ndim != 3
            or controls.shape[-1] != self.model.domain.control_dimension
            or controls.shape[1] < 1
            or controls.shape[0] < 1
        ):
            raise ValueError("controls must be (H,A) or (candidate,H,A) domain controls")
        initial_states = jnp.broadcast_to(
            jnp.asarray(initial_state, dtype=self.coefficients.dtype),
            (len(controls), self.model.domain.state_dimension),
        )
        return _rollouts(self.model, self.coefficients, initial_states, controls)

    def log_likelihood(self, transitions):
        _check_interval(self.model, transitions.timestep)
        states, controls, targets = _arrays(self.model, transitions)
        b, basis = components(self.model, states, controls)
        y = self.model.targets(states, controls, targets)
        prediction = b[None] + jnp.einsum(
            "nkd,sd->snk", basis, self.coefficients, precision=jax.lax.Precision.HIGHEST
        )
        _, variance = prior_and_variance(self.model.parameters, self.model.config)
        density = -0.5 * jnp.sum(
            (y[None] - prediction) ** 2 / variance + jnp.log(2 * jnp.pi * variance), axis=(1, 2)
        )
        return density - len(transitions) * jnp.log(self.model.normalizer.target_scale).sum()


@dataclass(frozen=True)
class Posterior:
    model: PopulationModel
    state: object
    seen_ids: frozenset[str] = frozenset()
    generation: int = 0

    def moments(self):
        if not self.seen_ids and self.generation == 0:
            mean, factor, _ = prior_moments_and_variance(self.model.parameters, self.model.config)
            return mean, jnp.matmul(factor, factor.T, precision=jax.lax.Precision.HIGHEST)
        return self.state.moments()

    def condition(self, transitions):
        _check_interval(self.model, transitions.timestep)
        duplicate = self.seen_ids.intersection(transitions.ids)
        if duplicate:
            raise ValueError(
                "transition already conditioned; rebuild from prior to replay evidence"
            )
        if not len(transitions):
            return self
        states, controls, targets = _arrays(self.model, transitions)
        b, basis = components(self.model, states, controls)
        y = self.model.targets(states, controls, targets)
        _, variance = prior_and_variance(self.model.parameters, self.model.config)
        state = gaussian_condition(
            self.state, basis, y - b, variance, jnp.ones(len(transitions), dtype=states.dtype)
        )
        if not np.isfinite(jax.device_get(state.precision)).all():
            raise FloatingPointError("nonfinite posterior update")
        return Posterior(
            self.model, state, self.seen_ids.union(transitions.ids), self.generation + 1
        )

    def predict(self, state, physical_control):
        dtype = self.model.normalizer.input_mean.dtype
        states = jnp.asarray(state, dtype=dtype)
        controls = jnp.asarray(physical_control, dtype=dtype)
        if states.shape[-1] != self.model.domain.state_dimension or controls.shape != (
            *states.shape[:-1],
            self.model.domain.control_dimension,
        ):
            raise ValueError("predict expects domain state (...,D) and matching control (...,A)")
        if not self.seen_ids and self.generation == 0:
            return _predict_prior(self.model, states, controls)
        return _predict(self.model, self.state, states, controls)

    def sample_functions(self, key, count):
        if type(count) is not int or count < 1:
            raise ValueError("sample count must be a positive integer")
        if not self.seen_ids and self.generation == 0:
            return self.model.sample_functions(key, count)
        mean, covariance = self.state.moments()
        coefficients = mean + jnp.matmul(
            jax.random.normal(key, (count, len(mean)), dtype=mean.dtype),
            jnp.linalg.cholesky(covariance).T,
            precision=jax.lax.Precision.HIGHEST,
        )
        return FunctionSamples(self.model, coefficients)

    def fantasize(self, state, physical_control, key, include_observation_noise=True):
        function_key, noise_key = jax.random.split(key)
        sample = self.sample_functions(function_key, 1)
        dtype = sample.coefficients.dtype
        state, control = jnp.asarray(state, dtype=dtype), jnp.asarray(physical_control, dtype=dtype)
        target = sample.step(state, control)[0]
        if include_observation_noise:
            _, variance = prior_and_variance(self.model.parameters, self.model.config)
            delta = increments(state, target, self.model.config.domain) + jax.random.normal(
                noise_key, (self.model.domain.state_dimension,), dtype=dtype
            ) * (jnp.sqrt(variance) * self.model.normalizer.target_scale)
            target = reconstruct(state, delta, self.model.config.domain)
        identifier = f"fantasy:{np.asarray(jax.random.key_data(key)).tolist()}:{self.generation}"
        transitions = Transitions(
            np.asarray(state)[None],
            np.asarray(control)[None],
            np.asarray(target)[None],
            (identifier,),
            self.model.config.timestep,
            self.model.config.domain,
        )
        return target, self.condition(transitions)

    def save(self, path):
        from dyna_dist.io.checkpoint import save_posterior

        save_posterior(self, path)


def _check_interval(model, timestep):
    if not math.isclose(timestep, model.config.timestep, rel_tol=1e-9):
        raise ValueError("transition interval differs from trained model timestep")


def _arrays(model, transitions):
    if transitions.domain != model.config.domain:
        raise ValueError("transition domain differs from learned model")
    dtype = model.normalizer.input_mean.dtype
    return tuple(
        jnp.asarray(x, dtype=dtype)
        for x in (transitions.states, transitions.controls, transitions.next_states)
    )
