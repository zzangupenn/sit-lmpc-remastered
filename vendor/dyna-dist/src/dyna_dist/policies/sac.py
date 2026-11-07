"""Exact deterministic SB3 SAC ReLU/tanh actor export, not a learned dynamics model."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from dyna_dist.domains import get_domain


@jax.jit
def _action(weights, biases, observation, low, high):
    x = observation
    for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
        x = jax.nn.relu(jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST) + bias)
    mean = jnp.matmul(x, weights[-1].T, precision=jax.lax.Precision.HIGHEST) + biases[-1]
    return low + (jnp.tanh(mean) + 1) * (high - low) / 2


@partial(jax.jit, static_argnames=("steps",))
def _closed_loop(model, coefficients, weights, biases, low, high, state, steps):
    from dyna_dist.inference.posterior import _particle_steps

    state = jnp.broadcast_to(state, (len(coefficients), 18))

    def advance(x, _):
        u = _action(weights, biases, jnp.asarray(x[..., 1:], dtype=jnp.float32), low, high)
        target = _particle_steps(model, coefficients, x, u)
        return target, (target, u)

    states, controls = jax.lax.scan(advance, state, None, length=steps)[1]
    return states.swapaxes(0, 1), controls.swapaxes(0, 1)


@dataclass(frozen=True)
class JaxSACPolicy:
    weights: tuple
    biases: tuple
    low: object
    high: object
    metadata: object

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("schema_version") != 1 or manifest.get("kind") != "halfcheetah_sac_policy":
            raise ValueError("unsupported warm-up policy artifact")
        if manifest["normalization"] != "identity" or manifest["activation"] != "relu":
            raise ValueError("policy requires unsupported preprocessing/activation")
        payload = (path / "actor.npz").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["checksums"]["actor.npz"]:
            raise ValueError("policy actor checksum mismatch")
        if tuple(manifest["state_fields"]) != get_domain("halfcheetah").state_fields:
            raise ValueError("policy physical-state ordering mismatch")
        with np.load(path / "actor.npz", allow_pickle=False) as data:
            weights = tuple(
                jnp.asarray(data[f"weight_{i}"], dtype=jnp.float32)
                for i in range(manifest["layers"])
            )
            biases = tuple(
                jnp.asarray(data[f"bias_{i}"], dtype=jnp.float32) for i in range(manifest["layers"])
            )
        if weights[0].shape[1] != 17 or weights[-1].shape[0] != 6:
            raise ValueError("HalfCheetah actor dimensions mismatch")
        return cls(
            weights,
            biases,
            jnp.asarray(manifest["action_low"], dtype=jnp.float32),
            jnp.asarray(manifest["action_high"], dtype=jnp.float32),
            manifest,
        )

    def observation_action(self, observation):
        observation = jnp.asarray(observation, dtype=jnp.float32)
        if observation.shape[-1] != 17:
            raise ValueError("SAC policy expects 17 raw observation components")
        return _action(self.weights, self.biases, observation, self.low, self.high)

    def action(self, physical_state):
        state = jnp.asarray(physical_state, dtype=jnp.float32)
        if state.shape[-1] != 18:
            raise ValueError("HalfCheetah policy expects full physical state (...,18)")
        return self.observation_action(state[..., 1:])

    def rollout(self, functions, initial_state, horizon):
        """Closed-loop trajectories with each sampled function's own state feedback."""
        if (
            functions.model.config.domain != "halfcheetah"
            or type(horizon) is not int
            or horizon < 1
        ):
            raise ValueError(
                "closed-loop policy rollout needs a HalfCheetah model and positive horizon"
            )
        return _closed_loop(
            functions.model,
            functions.coefficients,
            self.weights,
            self.biases,
            self.low,
            self.high,
            jnp.asarray(initial_state, dtype=functions.coefficients.dtype),
            horizon,
        )
