"""Joint function likelihood, with optional historical context adaptation."""

import jax
import jax.numpy as jnp
from dyna_dist.models.features import network, prior_and_variance, prior_moments_and_variance
from dyna_dist.models.gaussian import (
    condition, joint_nll, joint_nll_from_cholesky, joint_nll_from_qr,
)


def predictive_loss(parameters, batch, config, objective="context_predictive", stable=False):
    if objective == "population_joint":
        prior_mean, prior_factor, variance = prior_moments_and_variance(parameters, config)
    else:
        prior, variance = prior_and_variance(parameters, config)
    net = network(config)

    if objective == "population_joint":
        def instance(qx, qy):
            qb, qa = net.apply({"params": parameters["network"]}, qx)
            if not config.bayesian:
                return 0.5 * jnp.mean(
                    (qy - qb) ** 2 / variance + jnp.log(2 * jnp.pi * variance)
                )
            likelihood = joint_nll_from_qr if stable else joint_nll_from_cholesky
            return likelihood(
                prior_mean, prior_factor, qa, qy - qb,
                variance, jnp.ones(len(qx), dtype=qx.dtype)
            ) / (qy.shape[-1] * len(qx))

        return jax.vmap(instance)(batch["query_inputs"], batch["query_targets"]).mean()
    if objective != "context_predictive":
        raise ValueError(f"unknown training objective: {objective}")

    def instance(cx, cy, mask, qx, qy):
        cb, ca = net.apply({"params": parameters["network"]}, cx)
        qb, qa = net.apply({"params": parameters["network"]}, qx)
        if not config.bayesian:
            # Zero-basis case is independent Gaussian regression. Its closed form
            # avoids needless latent solves and GPU autodiff through zero matrices.
            return 0.5 * jnp.mean((qy - qb) ** 2 / variance + jnp.log(2 * jnp.pi * variance))
        posterior = condition(prior, ca, cy - cb, variance, mask)
        return joint_nll(posterior, qa, qy - qb, variance, jnp.ones(len(qx), dtype=qx.dtype)) / (
            qy.shape[-1] * len(qx)
        )

    losses = jax.vmap(instance)(
        batch["context_inputs"],
        batch["context_targets"],
        batch["context_mask"],
        batch["query_inputs"],
        batch["query_targets"],
    )
    return losses.mean()


def expected_complete_loss(parameters, batch, config):
    """Neural/noise M-step with one frozen posterior per complete vehicle instance.

    Within each instance a uniform minibatch estimates its full-data average
    expected log likelihood. The latent KL is constant in this M-step: the prior
    is updated separately from posterior sufficient statistics, not by SGD.
    """
    _, _, variance = prior_moments_and_variance(parameters, config)
    net = network(config)

    def instance(x, y, mean, covariance, weight):
        b, a = net.apply({"params": parameters["network"]}, x)
        error = y - b - jnp.einsum("nkd,d->nk", a, mean)
        latent_variance = jnp.einsum("nkd,de,nke->nk", a, covariance, a)
        return weight * 0.5 * jnp.mean(
            (error**2 + latent_variance) / variance + jnp.log(2 * jnp.pi * variance)
        )

    return jax.vmap(instance)(
        batch["query_inputs"], batch["query_targets"],
        batch["latent_mean"], batch["latent_covariance"],
        batch["likelihood_weight"],
    ).mean()
