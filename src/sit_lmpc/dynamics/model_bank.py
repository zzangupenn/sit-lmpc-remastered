"""Four-or-more fixed scenarios fitted by energy score with a mean-error budget.

These are predictive-distribution weights, not a Bayesian friction posterior.
The same vector supplies the ensemble mean and exact finite-bank event probabilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .belief import BeliefSnapshot
from .ensemble_fit import EnsembleFitBackend, blend_loss_and_grad


@jax.jit
def energy_statistics(predictions, targets, mask, coefficients, scales, supported):
    """Weighted full-window distances; angular chord embeddings handle wraparound.

    RMS normalization prevents long windows from receiving extra evidence mass.
    Unlike squared-distance energy, ordinary Euclidean energy scores distributions,
    not only their means. Invalid/padded/zero-prior branches never contribute NaNs.
    """
    predictions = jnp.where(supported[:, None, None, None] & mask[None, ..., None],
                            predictions, 0)
    targets = jnp.where(mask[..., None], targets, 0)

    def embed(states):
        parts = []
        for field in range(7):
            if field in (2, 4, 6):
                parts.extend((jnp.sin(states[..., field]) / scales[field],
                              jnp.cos(states[..., field]) / scales[field]))
            else:
                parts.append(states[..., field] / scales[field])
        return jnp.stack(parts, axis=-1)

    members, truth = embed(predictions), embed(targets)
    denominator = jnp.maximum(mask.sum(axis=-1), 1) * 7

    def distance(delta):
        squared = jnp.sum(jnp.where(mask[..., None], delta**2, 0), axis=(-2, -1))
        return jnp.sqrt(jnp.maximum(squared / denominator, 0))

    to_truth = jax.vmap(lambda member: distance(member - truth))(members)
    pairwise = jax.vmap(lambda a: jax.vmap(lambda b: distance(a - b))(members))(members)
    return to_truth @ coefficients, jnp.einsum("ijn,n->ij", pairwise, coefficients)


class ModelBankBackend(EnsembleFitBackend):
    method = "mean_constrained_energy_v1"

    def __init__(self, bank, config, horizon=None):
        super().__init__(bank, config, config.fit_horizon or horizon or 10)

    def _energy(self, cached):
        stats = [energy_statistics(*batch, self.scales, jnp.asarray(self.prior > 0))
                 for batch in cached]
        a, distances = jax.device_get((sum(s[0] for s in stats), sum(s[1] for s in stats)))
        return np.asarray(a, dtype=float), np.asarray(distances, dtype=float)

    def warmup(self, state):
        super().warmup(state)
        # Compile the energy kernel at the same padded horizon/batch shape as fitting.
        from sit_lmpc.data.dynamics import RolloutBatch

        windows = RolloutBatch(
            np.asarray(state)[None], np.zeros((1, self.horizon, 2)),
            np.broadcast_to(state, (1, self.horizon, 7)),
            np.full((1, self.horizon), self.bank.timestep),
            np.full((1, self.horizon), self.bank.integrator_timestep),
            np.ones((1, self.horizon), dtype=bool),
        )
        self._energy(self._cache(windows, np.ones(1)))

    def fit(self, selection, generation):
        if not len(selection.transitions):
            return None, {"status": "no_valid_transitions"}
        if selection.windows is None or selection.windows.controls.shape[1] != self.horizon:
            raise ValueError("model_bank requires recorded rollout windows matching fit_horizon")
        try:
            cached = self._cache(selection.windows, selection.weights)
        except FloatingPointError:
            return None, {"status": "nonfinite_model_rollout", "horizon": self.horizon}

        reference, regression = self._fit_mean(cached)
        reference_metrics = self._metrics(reference, cached)
        if (not np.isfinite(regression["loss"])
                or reference_metrics["ensemble_standardized_rmse"] > self.config.max_standardized_rmse):
            return None, {**reference_metrics, "status": "ensemble_mismatch"}
        a, distances = self._energy(cached)
        if not np.isfinite(a).all() or not np.isfinite(distances).all():
            return None, {"status": "nonfinite_energy_score"}

        last_mean_key, last_mean_value = None, None

        def mean_loss(w):
            nonlocal last_mean_key, last_mean_value
            key = np.asarray(w, dtype=self.bank.params.dtype).tobytes()
            if key == last_mean_key:
                return last_mean_value
            values = [blend_loss_and_grad(jnp.asarray(w, dtype=self.bank.params.dtype),
                                         *batch, self.scales) for batch in cached]
            value, grad = jax.device_get((sum(v[0] for v in values), sum(v[1] for v in values)))
            last_mean_key = key
            last_mean_value = float(value), np.asarray(grad, dtype=float)
            return last_mean_value

        def energy(w):
            return float(w @ a - 0.5 * w @ distances @ w), a - distances @ w

        # Use the reference after casting to actual planner precision. Every candidate
        # is checked in that same precision before publication, including renormalization.
        def canonical(w):
            w = np.where(self.prior > 0, np.maximum(w, 0), 0)
            return np.asarray(w / w.sum(), dtype=self.bank.params.dtype).astype(float)

        reference = canonical(reference)
        reference_loss, _ = mean_loss(reference)
        limit = reference_loss + self.config.mean_fit_slack
        best, best_score = reference.copy(), energy(reference)[0]
        initial_energy = best_score
        tolerance = float(8 * np.finfo(np.dtype(self.bank.params.dtype)).eps * max(1., abs(limit)))

        def retain(w):
            nonlocal best, best_score
            if (not np.isfinite(w).all() or np.min(w) < -1e-8
                    or abs(w.sum() - 1) > 1e-7 or np.any(w[self.prior == 0] != 0)):
                return
            candidate = canonical(w)
            loss, _ = mean_loss(candidate)
            score, _ = energy(candidate)
            if np.isfinite(loss) and loss <= limit + tolerance and score < best_score:
                best, best_score = candidate.copy(), score

        # Energy statistics are cached once; only the mean-error constraint evaluates
        # JAX gradients during SLSQP. No ST rollouts occur inside the optimizer.
        def objective(w):
            retain(w)
            return energy(w)

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1,
             "jac": lambda w: np.ones_like(w)},
            {"type": "ineq", "fun": lambda w: limit - mean_loss(w)[0],
             "jac": lambda w: -mean_loss(w)[1]},
        ]
        if np.count_nonzero(self.prior) == 1:
            iterations, converged, message = 0, True, "single supported model"
        else:
            result = minimize(
                objective, reference, jac=True, method="SLSQP",
                bounds=[(0, 1) if prior > 0 else (0, 0) for prior in self.prior],
                constraints=constraints,
                options={"maxiter": self.config.fit_max_iterations,
                         "ftol": self.config.fit_tolerance},
            )
            retain(result.x)
            iterations, converged, message = int(result.nit), bool(result.success), str(result.message)

        published = jnp.asarray(best, dtype=self.bank.params.dtype)
        final_loss, _ = mean_loss(np.asarray(published, dtype=float))
        metrics = {
            **self._metrics(best, cached),
            "status": "ready", "weight_kind": "model_probabilities",
            "probability_source": "fitted_predictive_distribution", "method": self.method,
            "horizon": self.horizon, "windows": len(selection.windows),
            "initial_loss": regression["initial_loss"], "loss": final_loss,
            "reference_loss": reference_loss, "reference_weights": reference.tolist(),
            "mean_fit_slack": self.config.mean_fit_slack, "mean_loss_limit": limit,
            "constraint_tolerance": tolerance,
            "mean_constraint_satisfied": bool(final_loss <= limit + tolerance),
            "initial_energy_score": initial_energy, "energy_score": best_score,
            "regression": regression, "optimizer_iterations": iterations,
            "optimizer_converged": converged, "optimizer_message": message,
            "used_reference": bool(np.array_equal(best, reference)),
        }
        if (not metrics["mean_constraint_satisfied"] or not np.isfinite(final_loss)
                or metrics["ensemble_standardized_rmse"] > self.config.max_standardized_rmse):
            return None, {**metrics, "status": "distribution_mismatch"}
        self.weights = np.array(published, dtype=float, copy=True)
        self.weights /= self.weights.sum()
        return BeliefSnapshot(self.bank, published, generation, "model_bank"), metrics
