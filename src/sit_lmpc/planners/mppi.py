"""Nominal dynamic-ST warm-up/tracking with explicit functional JAX kernels."""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from sit_lmpc.dynamics.ensemble import (
    STEnsemble, predictor_rollout as rollout, rollout_prediction,
)
from .common import PlannerResult, returns, weights
from .reference import nearest_point_jax, get_reference_trajectory_jax


@partial(jax.jit, static_argnames=("config",))
def iteration(model, initial, previous_actions, reference, key, config):
    da = jax.random.truncated_normal(
        key, -1 - previous_actions, 1 - previous_actions, shape=(config.samples, config.horizon, 2),
        dtype=model.params.dtype,
    )
    actions = jnp.clip(previous_actions[None] + da, -1, 1)
    states = jax.vmap(lambda a: rollout(model, initial, a, config.scan_rollout))(actions)
    reward = -jnp.abs(states[:, :, :2] - reference[None, 1:, :2]).sum(axis=-1)
    w = weights(returns(reward), config.temperature, config.damping)
    optimal = jnp.clip(previous_actions + jnp.sum(w[:, :, None] * da, axis=0), -1, 1)
    prediction, members = rollout_prediction(model, initial, optimal, config.scan_rollout)
    return optimal, prediction, members if hasattr(model, "weights") else None


@partial(jax.jit, static_argnames=("horizon",))
def reference_trajectory(state, target_speed, waypoints, diff, distances, timestep, horizon):
    _, distance, _, _, index = nearest_point_jax(state[:2], waypoints[:, 1:3], diff)
    reference = get_reference_trajectory_jax(
        jnp.full(horizon, target_speed, dtype=state.dtype), distance, index,
        waypoints, horizon, distances, timestep,
    )
    adjusted = jnp.where(reference[:, 3] - state[4] > 5,
                         reference[:, 3] - 2 * jnp.pi, reference[:, 3])
    adjusted = jnp.where(adjusted - state[4] < -5, adjusted + 2 * jnp.pi, adjusted)
    return reference.at[:, 3].set(adjusted)


@partial(jax.jit, static_argnames=("config",))
def tracking_control(model, state, previous, waypoints, diff, distances, target_speed, keys, config):
    """One compiled tracking update; explicit keys preserve the old draw sequence."""
    reference = reference_trajectory(state, target_speed, waypoints, diff, distances,
                                     model.timestep, config.horizon)
    previous = jnp.concatenate((previous[1:], jnp.zeros_like(previous[:1])))

    def advance(actions, key):
        optimal, prediction, members = iteration(model, state, actions, reference, key, config)
        return optimal, (prediction, members)

    optimal, (predictions, members) = jax.lax.scan(advance, previous, keys)
    return optimal, predictions[-1], reference, None if members is None else members[-1]


class MPPI:
    def __init__(self, config, model, track, seed=0, randomness=None):
        self.config, self.model, self.track = config, model, track
        self.key = jax.random.key(seed)
        self.randomness = randomness
        self.actions = jnp.zeros((config.horizon, 2), dtype=model.params.dtype)
        self.waypoints = jnp.asarray(track.raceline.waypoints, dtype=model.params.dtype)
        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.distances = jnp.linalg.norm(self.diff, axis=1)

    def reference(self, state, target_speed):
        return reference_trajectory(jnp.asarray(state, dtype=self.model.params.dtype), target_speed,
                                    self.waypoints, self.diff, self.distances,
                                    self.model.timestep, self.config.horizon)

    def warmup(self, state, target_speed):
        keys = jax.random.split(self.key, self.config.iterations)
        jax.block_until_ready(tracking_control(
            self.model, jnp.asarray(state, dtype=self.model.params.dtype), self.actions,
            self.waypoints, self.diff, self.distances, target_speed, keys, self.config,
        ))

    def update(self, state, target_speed):
        if hasattr(self.model, "resample"):
            if self.randomness is None:
                self.key, dynamics_key = jax.random.split(self.key)
            else:
                dynamics_key = self.randomness.next_key()
            self.model = self.model.resample(dynamics_key)
        state = jnp.asarray(state, dtype=self.model.params.dtype)
        keys = []
        for _ in range(self.config.iterations):
            if self.randomness is None:
                self.key, key = jax.random.split(self.key)
            else:
                key = self.randomness.next_key()
            keys.append(key)
        self.actions, trajectory, reference, model_trajectories = tracking_control(
            self.model, state, self.actions, self.waypoints, self.diff, self.distances,
            target_speed, jnp.stack(keys), self.config,
        )
        action, trajectory, reference, model_trajectories, probabilities = jax.device_get(
            (self.actions[0], trajectory, reference, model_trajectories,
             self.model.weights if hasattr(self.model, "weights") else None)
        )
        if not np.isfinite(action).all() or not np.isfinite(trajectory).all():
            raise FloatingPointError("nonfinite MPPI action or rollout")
        return PlannerResult(action, trajectory, reference,
                             model_trajectories=model_trajectories, model_weights=probabilities)
