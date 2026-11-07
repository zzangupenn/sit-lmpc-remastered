"""Pose-invariant observables; no physical parameter or instance-ID input."""

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from dyna_dist.domains import get_domain


@jax.jit
def wrap(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


@partial(jax.jit, static_argnames=("domain",))
def features(states, controls, domain="st"):
    if domain in ("halfcheetah", "walker2d", "dm_walker"):
        return jnp.concatenate((states[..., 1:], controls), axis=-1)
    return jnp.concatenate((states[..., jnp.array([2, 3, 5, 6])], controls), axis=-1)


@partial(jax.jit, static_argnames=("domain",))
def increments(states, targets, domain="st"):
    delta = targets - states
    if domain in ("halfcheetah", "walker2d", "dm_walker"):
        return delta
    c, s = jnp.cos(states[..., 4]), jnp.sin(states[..., 4])
    local_x = c * delta[..., 0] + s * delta[..., 1]
    local_y = -s * delta[..., 0] + c * delta[..., 1]
    return (
        delta.at[..., 0]
        .set(local_x)
        .at[..., 1]
        .set(local_y)
        .at[..., 4]
        .set(wrap(delta[..., 4]))
        .at[..., 6]
        .set(wrap(delta[..., 6]))
    )


@partial(jax.jit, static_argnames=("domain",))
def reconstruct(states, delta, domain="st"):
    if domain in ("halfcheetah", "walker2d", "dm_walker"):
        return states + delta
    c, s = jnp.cos(states[..., 4]), jnp.sin(states[..., 4])
    global_delta = (
        delta.at[..., 0]
        .set(c * delta[..., 0] - s * delta[..., 1])
        .at[..., 1]
        .set(s * delta[..., 0] + c * delta[..., 1])
    )
    result = states + global_delta
    return result.at[..., 4].set(wrap(result[..., 4])).at[..., 6].set(wrap(result[..., 6]))


@partial(jax.jit, static_argnames=("domain",))
def global_covariance(states, covariance, domain="st"):
    if domain in ("halfcheetah", "walker2d", "dm_walker"):
        return covariance
    rotation = jnp.broadcast_to(jnp.eye(7, dtype=states.dtype), (*states.shape[:-1], 7, 7))
    c, s = jnp.cos(states[..., 4]), jnp.sin(states[..., 4])
    rotation = (
        rotation.at[..., 0, 0]
        .set(c)
        .at[..., 0, 1]
        .set(-s)
        .at[..., 1, 0]
        .set(s)
        .at[..., 1, 1]
        .set(c)
    )
    return jnp.matmul(
        jnp.matmul(rotation, covariance, precision=jax.lax.Precision.HIGHEST),
        jnp.swapaxes(rotation, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )


@struct.dataclass
class Normalizer:
    input_mean: jax.Array
    input_scale: jax.Array
    target_mean: jax.Array
    target_scale: jax.Array

    @classmethod
    def fit(cls, inputs, targets, dtype, domain="st"):
        # Small but nonzero scales protect deterministic/constant target components.
        spec = get_domain(domain)
        input_floor = np.array(spec.input_floor)
        target_floor = np.array(spec.target_floor)
        if not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError("normalizers require nonempty finite training arrays")
        return cls(
            *(
                jnp.asarray(x, dtype=dtype)
                for x in (
                    inputs.mean(axis=0),
                    np.maximum(inputs.std(axis=0), input_floor),
                    targets.mean(axis=0),
                    np.maximum(targets.std(axis=0), target_floor),
                )
            )
        )

    def normalize_inputs(self, inputs):
        return (inputs - self.input_mean) / self.input_scale

    def normalize_targets(self, targets):
        return (targets - self.target_mean) / self.target_scale

    def denormalize_targets(self, targets):
        return targets * self.target_scale + self.target_mean
