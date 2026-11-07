"""Stateless, compiled temporary Gaussian updates; never publish real evidence.

The current neural feature map, normalization and likelihood must remain fixed.
Unlike the CPU float64 streaming ledger, these kernels accept arrays and return
new moments/factors, making them suitable for batched hypothetical learning.
Precision is the input precision: callers must not silently downcast evidence.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from .posterior import components


@jax.jit
def condition_factor(mean, factor, basis, residual, variance, mask, power=1.0):
    """Condition a Gaussian factor with masked observations in prior coordinates.

    basis: (N,K,D), residual: (N,K), factor: (D,D), mask: (N,).
    Returns (new_mean, new_factor, valid). A false validity flag must not be
    interpreted as a usable posterior. Masked padding is integrated out even if
    its placeholders are nonfinite. Likelihood power 1 is conditional Bayes.
    """
    if (factor.shape != (len(mean), len(mean))
            or basis.shape != (*residual.shape, len(mean))
            or variance.shape != residual.shape[1:] or mask.shape != residual.shape[:1]):
        raise ValueError("Gaussian conditioner dimensions differ")
    active = mask > 0
    basis = jnp.where(active[:, None, None], basis, 0)
    residual = jnp.where(active[:, None], residual, 0)
    scale = jnp.sqrt(jnp.where(active, mask, 0)[:, None] * power / variance[None])
    matrix = jnp.matmul(basis, factor, precision=jax.lax.Precision.HIGHEST) * scale[..., None]
    centered = (residual - jnp.einsum("nkd,d->nk", basis, mean,
                                      precision=jax.lax.Precision.HIGHEST)) * scale
    matrix = matrix.reshape(-1, len(mean))
    centered = centered.reshape(-1)
    gram = jnp.eye(len(mean), dtype=mean.dtype) + jnp.matmul(
        matrix.T, matrix, precision=jax.lax.Precision.HIGHEST)
    information = jnp.matmul(matrix.T, centered, precision=jax.lax.Precision.HIGHEST)
    chol = jnp.linalg.cholesky((gram + gram.T) / 2)
    white_mean = cho_solve((chol, True), information)
    updated_mean = mean + jnp.matmul(factor, white_mean, precision=jax.lax.Precision.HIGHEST)
    updated_factor = solve_triangular(chol, factor.T, lower=True).T
    valid = (jnp.all(jnp.isfinite(updated_mean)) & jnp.all(jnp.isfinite(updated_factor))
             & jnp.all(jnp.isfinite(variance) & (variance > 0))
             & jnp.all(jnp.isfinite(mask) & (mask >= 0) & (mask <= 1))
             & jnp.isfinite(power) & (power > 0) & (power <= 1))
    # Preserve an uninformative update exactly, including the caller's factor convention.
    empty = ~jnp.any(active)
    return jnp.where(empty, mean, updated_mean), jnp.where(empty, factor, updated_factor), valid


@partial(jax.jit, static_argnames=("chunk_size",))
def condition_factor_batch(mean, factor, basis, residual, variance, mask, power=1.0, *, chunk_size=8):
    """One shared current belief, independent hypothetical branches; no mutation.

    basis/residual/mask have a leading branch axis. Device-side chunking bounds
    solve workspace. Returned factors still require O(branches * D**2) storage;
    a future regret kernel should reduce branch costs before retaining all of it.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return jax.lax.map(
        lambda observation: condition_factor(mean, factor, *observation[:2], variance,
                                              observation[2], power),
        (basis, residual, mask), batch_size=chunk_size,
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def condition_transitions(model, mean, factor, states, controls, next_states, mask,
                          power=1.0, *, chunk_size=8):
    """Feature extraction and conditioning for (branches, transitions, state/action)."""
    from dyna_dist.models.features import prior_moments_and_variance

    if states.ndim != 3 or states.shape != next_states.shape or mask.shape != states.shape[:2]:
        raise ValueError("hypothetical transitions must be (branches, transitions, state)")
    active = mask > 0
    states = jnp.where(active[..., None], states, 0)
    controls = jnp.where(active[..., None], controls, 0)
    next_states = jnp.where(active[..., None], next_states, 0)
    base, basis = components(model, states, controls)
    residual = model.targets(states, controls, next_states) - base
    _, _, variance = prior_moments_and_variance(model.parameters, model.config)
    return condition_factor_batch(mean, factor, basis, residual, variance, mask, power,
                                   chunk_size=chunk_size)
