"""Train-only latent inference, population EM moments, and canonical whitening."""

from functools import partial

from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from dyna_dist.models.features import network, prior_moments_and_variance


@partial(jax.jit, static_argnames=("config",))
def _statistics(parameters, x, y, mask, mean, factor, variance, config):
    b, a = network(config).apply({"params": parameters["network"]}, x)
    a = a.astype(jnp.float64)
    scale = jnp.sqrt(mask[:, None] / variance[None])
    matrix = ((a @ factor) * scale[..., None]).reshape(-1, len(mean))
    residual = y.astype(jnp.float64) - b.astype(jnp.float64) - jnp.einsum("nkd,d->nk", a, mean)
    target = (residual * scale).reshape(-1)
    return matrix.T @ matrix, matrix.T @ target


def infer_latents(model, instances, count=0, seed=0, chunk_size=4096):
    """One posterior per physical instance; float64 sufficient-statistic solves.

    Network evaluation remains in its declared dtype. All transitions are used
    when count=0. Fixed-size padded chunks bound memory without changing the
    likelihood or counting padding as observations. No validation/test input is
    obtained here: the caller explicitly supplies the training instances.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("latent sufficient statistics require jax_enable_x64")
    mean, factor, variance = [np.asarray(value, np.float64) for value in
                             prior_moments_and_variance(model.parameters, model.config)]
    means, covariances, counts = [], [], []
    for index, (x, y, _, _) in enumerate(instances):
        if not len(x):
            raise ValueError("cannot infer a latent from an empty physics instance")
        if count and len(x) > count:
            order = np.random.default_rng(np.random.SeedSequence([seed, 917, index])).choice(
                len(x), count, replace=False
            )
            x, y = x[order], y[order]
        gram, information = np.eye(len(mean)), np.zeros(len(mean))
        width = min(chunk_size, len(x))
        for start in range(0, len(x), width):
            size = min(width, len(x) - start)
            xx = np.zeros((width, x.shape[-1]), dtype=x.dtype)
            yy = np.zeros((width, y.shape[-1]), dtype=y.dtype)
            mask = np.zeros(width, dtype=np.float64)
            xx[:size], yy[:size], mask[:size] = x[start:start + size], y[start:start + size], 1
            gg, hh = _statistics(model.parameters, xx, yy, mask, mean, factor, variance, model.config)
            gram += np.asarray(gg)
            information += np.asarray(hh)
        chol = cho_factor((gram + gram.T) / 2, lower=True)
        white_mean = cho_solve(chol, information)
        white_covariance = cho_solve(chol, np.eye(len(mean)))
        means.append(mean + factor @ white_mean)
        covariances.append(factor @ white_covariance @ factor.T)
        counts.append(len(x))
    return np.asarray(means), np.asarray(covariances), counts


def population_moments(means, covariances, floor):
    """EM M-step includes within-instance uncertainty, not only point latents."""
    mean = np.mean(means, axis=0)
    centered = means - mean
    covariance = np.mean(covariances, axis=0) + centered.T @ centered / len(means)
    covariance = (covariance + covariance.T) / 2
    factor = np.linalg.cholesky(covariance)
    diagonal = np.maximum(np.diag(factor), floor + 1e-7)
    factor[np.diag_indices_from(factor)] = diagonal
    return mean, factor


def with_prior(model, mean, factor):
    parameters = unfreeze(model.parameters)
    dtype = parameters["prior_mean"].dtype
    raw = np.tril(factor, -1) + np.diag(np.log(np.expm1(
        np.diag(factor) - model.config.prior_floor
    )))
    parameters["prior_mean"] = jnp.asarray(mean, dtype)
    parameters["prior_raw"] = jnp.asarray(raw, dtype)
    return model.replace(parameters=freeze(parameters))


def refit_population(model, instances, count=64, seed=0, iterations=1, chunk_size=4096):
    previous_precision = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        for _ in range(iterations):
            means, covariances, counts = infer_latents(model, instances, count, seed, chunk_size)
            mean, factor = population_moments(means, covariances, model.config.prior_floor)
            model = with_prior(model, mean, factor)
    finally:
        jax.config.update("jax_enable_x64", previous_precision)
    return model, {
        "method": "Gaussian posterior-moment EM update", "iterations": iterations,
        "instances": len(instances), "transitions_per_instance": counts,
        "fit_split": "train", "seed": seed,
        "mean": mean.tolist(), "covariance": (factor @ factor.T).tolist(),
        "includes_posterior_covariance": True,
    }


def canonicalize(model, means, covariances):
    """Absorb updated population moments into linear heads; prior becomes N(0,I).

    This is an exact affine reparameterization of the *updated* population, not
    a shrinkage heuristic. Individual posterior functions are preserved. Adam
    is restarted by the caller because its moments use the old coordinates.
    """
    mean, factor = population_moments(means, covariances, model.config.prior_floor)
    parameters = unfreeze(model.parameters)
    heads = parameters["network"]
    d = model.config.latent_dimension
    for key in ("kernel", "bias"):
        original = np.asarray(heads["basis"][key])
        basis = original.reshape(*original.shape[:-1], -1, d).astype(np.float64)
        b = np.asarray(heads["mean"][key], np.float64)
        heads["mean"][key] = jnp.asarray(b + basis @ mean, heads["mean"][key].dtype)
        heads["basis"][key] = jnp.asarray((basis @ factor).reshape(original.shape), original.dtype)
    identity_raw = np.log(np.expm1(1 - model.config.prior_floor))
    parameters["prior_mean"] = jnp.zeros_like(parameters["prior_mean"])
    parameters["prior_raw"] = jnp.eye(d, dtype=parameters["prior_raw"].dtype) * identity_raw
    transformed_means = solve_triangular(factor, (means - mean).T, lower=True).T
    inverse = solve_triangular(factor, np.eye(d), lower=True)
    # Contract pairwise: the default direct three-operand einsum is O(N*d^4),
    # whereas the identical inverse @ covariance @ inverse.T is O(N*d^3).
    transformed_covariances = np.einsum(
        "ij,njk,lk->nil", inverse, covariances, inverse, optimize=True
    )
    return model.replace(parameters=freeze(parameters)), transformed_means, transformed_covariances


def infer_and_canonicalize(model, instances, config):
    means, covariance, counts = infer_latents(
        model, instances, config.em_transitions, config.seed, config.em_chunk_size
    )
    model, means, covariance = canonicalize(model, means, covariance)
    return model, means, covariance, counts
