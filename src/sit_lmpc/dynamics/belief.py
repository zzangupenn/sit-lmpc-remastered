"""Backend-independent belief snapshots and deterministic physical likelihoods."""

from dataclasses import dataclass
from typing import Protocol
import jax
import jax.numpy as jnp
import numpy as np

from .bank import STModelBank
from .st import rk4


@dataclass(frozen=True)
class BeliefSnapshot:
    bank: STModelBank
    weights: jax.Array
    generation: int
    backend: str = "model_bank"
    posterior_weights: jax.Array | None = None

    def __post_init__(self):
        weights = np.asarray(self.weights)
        if (
            weights.shape != (len(self.bank.params),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError(
                "belief weights must be finite normalized probabilities matching model support"
            )
        if self.posterior_weights is not None:
            posterior = np.asarray(self.posterior_weights)
            if (posterior.shape != weights.shape or not np.isfinite(posterior).all()
                    or np.any(posterior < 0) or not np.isclose(posterior.sum(), 1)):
                raise ValueError("posterior weights must match and normalize over model support")

    @property
    def weight_kind(self):
        return "ensemble_coefficients" if self.backend == "ensemble_fit" else "model_probabilities"

    @property
    def categorical_weights(self):
        """Never use predictive blend coefficients as posterior probabilities."""
        return self.posterior_weights if self.backend == "ensemble_fit" else self.weights

    @property
    def probability_source(self):
        if self.generation == 0:
            return "categorical_prior"
        return "fitted_predictive_distribution" if self.backend == "model_bank" else "categorical_posterior"

    def representative(self):
        return self.bank.member(self.bank.params[int(np.argmax(np.asarray(self.weights)))])


class BeliefBackend(Protocol):
    prior: np.ndarray

    def fit(self, selection, generation): ...
    def score(self, transitions, weights): ...
    def warmup(self, state): ...


@jax.jit
def transition_statistics(params, states, controls, targets, intervals, substeps, scales, mask):
    """Fixed-shape bank x transition batches; controls are already physical."""

    def predict(model_params, state, control, interval, dt):
        count = jnp.rint(interval / dt).astype(jnp.int32)
        return jax.lax.fori_loop(0, count, lambda _, x: rk4(x, control, model_params, dt), state)

    predictions = jax.vmap(
        lambda model_params: jax.vmap(predict, in_axes=(None, 0, 0, 0, 0))(
            model_params, states, controls, intervals, substeps
        )
    )(params)
    residual = targets[None] - predictions
    for component in (2, 4, 6):
        angle = residual[..., component]
        residual = residual.at[..., component].set(jnp.arctan2(jnp.sin(angle), jnp.cos(angle)))
    standardized = residual / scales
    squared = jnp.mean(standardized**2, axis=-1)
    log_likelihood = -0.5 * jnp.sum(standardized**2 + jnp.log(2 * jnp.pi * scales**2), axis=-1)
    # Padding must be removed even when an unused prediction is nonfinite.
    return (
        jnp.where(mask[None], log_likelihood, 0),
        jnp.where(mask[None, :, None], residual**2, 0),
        jnp.where(mask[None], squared, 0),
    )


def replay_probabilities(prior, mean_log_likelihood, evidence_strength):
    prior = np.asarray(prior, dtype=np.float64)
    likelihood = np.asarray(mean_log_likelihood, dtype=np.float64)
    if (
        prior.shape != likelihood.shape
        or not np.isfinite(prior).all()
        or np.any(prior < 0)
        or prior.sum() <= 0
    ):
        raise ValueError("invalid dynamics belief prior")
    if not np.isfinite(likelihood).all():
        raise FloatingPointError("nonfinite model likelihood")
    log_prior = np.full_like(prior, -np.inf)
    positive = prior > 0
    log_prior[positive] = np.log(prior[positive] / prior.sum())
    # A shared log-density offset carries no model evidence. Remove it before
    # tempering to avoid overflow and preserve support under very strong evidence.
    log_weight = log_prior + evidence_strength * (likelihood - likelihood[positive].max())
    weights = np.exp(log_weight - log_weight.max())
    return weights / weights.sum()


class LikelihoodBackend:
    """Original tempered replay-likelihood method, retained as an explicit ablation."""

    method = "replay_likelihood_v1"
    def __init__(self, bank, config, *, horizon=None):
        self.bank, self.config = bank, config
        count = len(bank.params)
        self.prior = (
            np.ones(count) / count
            if config.prior is None
            else np.asarray(config.prior, dtype=float) / sum(config.prior)
        )
        self.scales = jnp.asarray(config.residual_scales, dtype=bank.params.dtype)

    def statistics(self, transitions, sample_weights=None):
        count = len(transitions)
        if not count:
            return None
        coefficients = (
            np.full(count, 1 / count) if sample_weights is None else np.asarray(sample_weights)
        )
        if (
            coefficients.shape != (count,)
            or not np.isfinite(coefficients).all()
            or np.any(coefficients < 0)
            or not np.isclose(coefficients.sum(), 1)
        ):
            raise ValueError("replay sample weights must normalize to one")
        model_count, size = len(self.bank.params), self.config.batch_size
        mean_log, mse, scaled_mse = (
            np.zeros(model_count),
            np.zeros((model_count, 7)),
            np.zeros(model_count),
        )
        dtype = self.bank.params.dtype
        for start in range(0, count, size):
            end, n = min(start + size, count), min(size, count - start)
            # Valid dummy state and intervals avoid undefined dynamics in padding.
            states, targets, controls = (
                np.zeros((size, 7)),
                np.zeros((size, 7)),
                np.zeros((size, 2)),
            )
            states[:, 3] = 1
            states[:n], targets[:n], controls[:n] = (
                transitions.states[start:end],
                transitions.next_states[start:end],
                transitions.controls[start:end],
            )
            intervals, substeps = np.ones(size) * 0.1, np.ones(size) * 0.02
            intervals[:n], substeps[:n] = (
                transitions.intervals[start:end],
                transitions.substeps[start:end],
            )
            log_likelihood, residual_squared, standardized_squared = jax.device_get(
                transition_statistics(
                    self.bank.params,
                    jnp.asarray(states, dtype=dtype),
                    jnp.asarray(controls, dtype=dtype),
                    jnp.asarray(targets, dtype=dtype),
                    jnp.asarray(intervals, dtype=dtype),
                    jnp.asarray(substeps, dtype=dtype),
                    self.scales,
                    jnp.asarray(np.arange(size) < n),
                )
            )
            weight = coefficients[start:end]
            mean_log += log_likelihood[:, :n] @ weight
            mse += np.einsum("mnd,n->md", residual_squared[:, :n], weight)
            scaled_mse += standardized_squared[:, :n] @ weight
        return {
            "mean_log_likelihood": mean_log,
            "component_rmse": np.sqrt(mse),
            "standardized_rmse": np.sqrt(scaled_mse),
        }

    def warmup(self, state):
        from sit_lmpc.data.dynamics import TransitionBatch

        self.statistics(
            TransitionBatch(
                np.asarray(state)[None],
                np.zeros((1, 2)),
                np.asarray(state)[None],
                np.asarray([self.bank.timestep]),
                np.asarray([self.bank.integrator_timestep]),
                np.zeros((1, 2), dtype=np.int64),
            )
        )

    def score(self, transitions, weights):
        stats = self.statistics(transitions)
        if stats is None:
            return {"status": "no_valid_transitions", "transitions": 0}
        probabilities = np.asarray(weights)
        return {
            "status": "pre_update",
            "transitions": len(transitions),
            "model_component_rmse": stats["component_rmse"].tolist(),
            "weighted_component_rmse": np.sqrt(
                probabilities @ stats["component_rmse"] ** 2
            ).tolist(),
        }

    def fit(self, selection, generation):
        stats = self.statistics(selection.transitions, selection.weights)
        if stats is None:
            return None, {"status": "no_valid_transitions"}
        metrics = {name: value.tolist() for name, value in stats.items()}
        supported = self.prior > 0
        finite = all(np.isfinite(value).all() for value in stats.values())
        if (
            not finite
            or np.min(stats["standardized_rmse"][supported]) > self.config.max_standardized_rmse
        ):
            return None, {**metrics, "status": "all_models_mismatch"}
        weights = replay_probabilities(
            self.prior, stats["mean_log_likelihood"], self.config.evidence_strength
        )
        return BeliefSnapshot(
            self.bank, jnp.asarray(weights, dtype=self.bank.params.dtype), generation, "likelihood"
        ), {**metrics, "status": "ready"}
