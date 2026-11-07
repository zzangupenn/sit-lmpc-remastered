"""Deterministic neural features with a compact joint-output Bayesian head."""

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from .gaussian import from_moments
from dyna_dist.domains import get_domain


class FeatureNetwork(nn.Module):
    hidden_sizes: tuple[int, ...]
    latent_dimension: int
    output_dimension: int = 7
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.hidden_sizes:
            x = nn.tanh(
                nn.Dense(
                    size,
                    dtype=self.dtype,
                    param_dtype=self.dtype,
                    precision=jax.lax.Precision.HIGHEST,
                )(x)
            )
        mean = nn.Dense(
            self.output_dimension,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=jax.lax.Precision.HIGHEST,
            name="mean",
        )(x)
        basis = nn.Dense(
            self.output_dimension * self.latent_dimension,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=jax.lax.Precision.HIGHEST,
            kernel_init=nn.initializers.normal(0.025),
            name="basis",
        )(x)
        return mean, basis.reshape(*inputs.shape[:-1], self.output_dimension, self.latent_dimension)


def network(config):
    return FeatureNetwork(
        config.hidden_sizes,
        config.latent_dimension,
        get_domain(config.domain).state_dimension,
        getattr(jnp, config.dtype),
    )


def initialize(config, key):
    dtype = getattr(jnp, config.dtype)
    d = config.latent_dimension
    spec = get_domain(config.domain)
    prior_diag = np.log(np.expm1(1 - config.prior_floor))
    noise_fraction = (config.noise_std - config.noise_floor) / (
        config.noise_ceiling - config.noise_floor
    )
    return {
        "network": network(config).init(key, jnp.zeros((1, spec.input_dimension), dtype=dtype))[
            "params"
        ],
        "prior_mean": jnp.zeros(d, dtype=dtype),
        "prior_raw": jnp.eye(d, dtype=dtype) * prior_diag,
        "noise_raw": jnp.full(
            spec.state_dimension, np.log(noise_fraction / (1 - noise_fraction)), dtype=dtype
        ),
    }


def prior_moments_and_variance(parameters, config):
    dtype = getattr(jnp, config.dtype)
    if config.learn_prior:
        raw = parameters["prior_raw"]
        factor = jnp.tril(raw, -1) + jnp.diag(jax.nn.softplus(jnp.diag(raw)) + config.prior_floor)
        mean = parameters["prior_mean"]
    else:
        mean = jnp.zeros(config.latent_dimension, dtype=dtype)
        factor = jnp.eye(config.latent_dimension, dtype=dtype)
    if config.learn_noise:
        std = config.noise_floor + (config.noise_ceiling - config.noise_floor) * jax.nn.sigmoid(
            parameters["noise_raw"]
        )
    else:
        std = jnp.full(get_domain(config.domain).state_dimension, config.noise_std, dtype=dtype)
    return mean, factor, std**2


def prior_and_variance(parameters, config):
    mean, factor, variance = prior_moments_and_variance(parameters, config)
    covariance = jnp.matmul(factor, factor.T, precision=jax.lax.Precision.HIGHEST)
    return from_moments(mean, covariance), variance
