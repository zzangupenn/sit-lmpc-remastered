from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PlannerResult:
    action: np.ndarray
    trajectory: np.ndarray
    reference: np.ndarray | None = None
    safe_set_violation: float = 0.0
    boundary_violation: float = 0.0
    lambdas: tuple[float, ...] = (0.0, 0.0)
    metrics: dict = field(default_factory=dict)
    debug: dict | None = None
    risk: dict | None = None
    model_trajectories: np.ndarray | None = None
    model_weights: np.ndarray | None = None


@jax.jit
def returns(reward):
    return jnp.flip(jnp.cumsum(jnp.flip(reward, axis=-1), axis=-1), axis=-1)


def weights(reward, temperature, damping, axis=-2, valid=None):
    if valid is None:
        maximum = jnp.max(reward, axis=axis, keepdims=True)
        minimum = jnp.min(reward, axis=axis, keepdims=True)
        standardized = (reward - maximum) / (maximum - minimum + damping)
    else:
        valid = jnp.broadcast_to(valid, reward.shape) & jnp.isfinite(reward)
        any_valid = jnp.any(valid, axis=axis, keepdims=True)
        maximum = jnp.max(jnp.where(valid, reward, -jnp.inf), axis=axis, keepdims=True)
        minimum = jnp.min(jnp.where(valid, reward, jnp.inf), axis=axis, keepdims=True)
        maximum, minimum = jnp.where(any_valid, maximum, 0.), jnp.where(any_valid, minimum, 0.)
        standardized = jnp.where(valid, (reward - maximum) / (maximum - minimum + damping), -jnp.inf)
    w = jnp.exp(standardized / temperature)
    return w / jnp.maximum(w.sum(axis=axis, keepdims=True), 1e-30)


def unit_range(values, valid=None):
    if valid is None:
        shifted = values - jnp.min(values)
    else:
        minimum = jnp.min(jnp.where(valid, values, jnp.inf))
        minimum = jnp.where(jnp.any(valid), minimum, 0.)
        shifted = jnp.where(valid, values - minimum, 0.)
    return shifted / jnp.maximum(jnp.max(shifted), 1e-12)


@partial(jax.jit, static_argnames=("track", "block_size"))
def to_frenet(track, states, search_indices, current_s, block_size=8192):
    shape = states.shape
    flat = states.reshape(-1, 7)
    poses = flat[:, (0, 1, 4)]
    if block_size and len(flat) > block_size:
        padded = jnp.pad(poses, ((0, (-len(flat)) % block_size), (0, 0)))
        converted = jax.lax.map(
            lambda block: track.vmap_cartesian_to_frenet_jax_jit(block, search_indices),
            padded.reshape(-1, block_size, 3),
        ).reshape(-1, 3)[:len(flat)]
    else:
        converted = track.vmap_cartesian_to_frenet_jax_jit(poses, search_indices)
    # Apply the existing global lift AFTER removing padding/combining blocks.
    progress = converted[:, 0]
    all_s = jnp.concatenate([progress, jnp.asarray(current_s).reshape(1)])
    finite = jnp.isfinite(all_s)
    center = (jnp.max(jnp.where(finite, all_s, -jnp.inf))
              + jnp.min(jnp.where(finite, all_s, jnp.inf))) / 2
    progress = progress + ((progress - center) < -track.s_frame_max / 3) * track.s_frame_max
    frenet = jnp.column_stack(
        [progress, converted[:, 1], flat[:, 2:4], converted[:, 2], flat[:, 5:7]]
    )
    return frenet.reshape(shape)
