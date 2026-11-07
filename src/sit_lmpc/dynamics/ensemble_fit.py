"""Fit simplex coefficients to cached, independent ST rollout predictions.

Coefficients minimize ensemble-mean error; separate categorical probabilities
retain the original likelihood backend's meaning for future distribution use.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from sit_lmpc.data.dynamics import RolloutBatch
from .belief import BeliefSnapshot, LikelihoodBackend
from .ensemble import weighted_states
from .st import rk4


@jax.jit
def window_predictions(params, states, controls, intervals, substeps, mask):
    """Physical controls, per-record integration context, and fixed friction."""

    def member(parameters):
        def window(initial, actions, durations, steps, valid):
            def advance(state, inputs):
                control, duration, dt, active = inputs
                count = jnp.rint(duration / dt).astype(jnp.int32)
                predicted = jax.lax.cond(
                    active,
                    lambda x: jax.lax.fori_loop(
                        0, count, lambda _, y: rk4(y, control, parameters, dt), x
                    ),
                    lambda x: x,
                    state,
                )
                return predicted, predicted

            return jax.lax.scan(advance, initial, (actions, durations, steps, valid))[1]

        return jax.vmap(window)(states, controls, intervals, substeps, mask)

    return jax.vmap(member)(params)


def blend_mse(weights, predictions, targets, mask, coefficients):
    blended = weighted_states(predictions, weights, differentiable=True)
    residual = jnp.where(mask[..., None], targets, 0) - jnp.where(mask[..., None], blended, 0)
    for field in (2, 4, 6):
        angle = residual[..., field]
        residual = residual.at[..., field].set(jnp.arctan2(jnp.sin(angle), jnp.cos(angle)))
    residual = jnp.where(mask[..., None], residual, 0)
    # A truncated window retains its start's evidence mass, not a fictitious
    # padded horizon or a lower weight merely because it ended in failure.
    per_window = jnp.sum(residual**2, axis=1) / jnp.maximum(mask.sum(axis=1), 1)[:, None]
    return jnp.einsum("nd,n->d", per_window, coefficients)


@jax.jit
def blend_statistics(weights, predictions, targets, mask, coefficients):
    mse = blend_mse(weights, predictions, targets, mask, coefficients)
    indices = jnp.maximum(mask.sum(axis=1) - 1, 0)
    rows = jnp.arange(len(targets))
    terminal = blend_mse(
        weights,
        predictions[:, rows, indices, :][:, :, None],
        targets[rows, indices][:, None],
        mask.any(axis=1)[:, None],
        coefficients,
    )
    return mse, terminal


@jax.jit
def blend_loss_and_grad(weights, predictions, targets, mask, coefficients, scales):
    def loss(w):
        return jnp.mean(blend_mse(w, predictions, targets, mask, coefficients) / scales**2)

    return jax.value_and_grad(loss)(weights)


class EnsembleFitBackend(LikelihoodBackend):
    method = "ensemble_regression_v1"

    def __init__(self, bank, config, horizon):
        super().__init__(bank, config)
        self.horizon = horizon
        self.weights = self.prior.copy()

    def _cache(self, windows, coefficients):
        batches = []
        size, dtype = self.config.batch_size, self.bank.params.dtype
        for start in range(0, len(windows), size):
            end = min(start + size, len(windows))
            count, horizon = end - start, windows.controls.shape[1]
            states = np.zeros((size, 7))
            states[:, 3] = 1
            states[:count] = windows.states[start:end]
            arrays = []
            for values, shape, fill in (
                (windows.controls, (size, horizon, 2), 0),
                (windows.targets, (size, horizon, 7), 0),
                (windows.intervals, (size, horizon), self.bank.timestep),
                (windows.substeps, (size, horizon), self.bank.integrator_timestep),
            ):
                padded = np.full(shape, fill, dtype=np.dtype(dtype))
                padded[:count] = values[start:end]
                arrays.append(jnp.asarray(padded, dtype=dtype))
            controls, targets, intervals, substeps = arrays
            mask = np.zeros((size, horizon), dtype=bool)
            mask[:count] = windows.mask[start:end]
            mask = jnp.asarray(mask)
            row_weights = np.zeros(size)
            row_weights[:count] = coefficients[start:end]
            predictions = window_predictions(
                self.bank.params,
                jnp.asarray(states, dtype=dtype),
                controls,
                intervals,
                substeps,
                mask,
            )
            finite = jnp.all(
                jnp.where(
                    jnp.asarray(self.prior > 0)[:, None, None, None] & mask[None, ..., None],
                    jnp.isfinite(predictions),
                    True,
                )
            )
            if not bool(finite):
                raise FloatingPointError("nonfinite supported model rollout")
            batches.append((predictions, targets, mask, jnp.asarray(row_weights, dtype=dtype)))
        return batches

    def _metrics(self, weights, cached):
        values = [
            blend_statistics(jnp.asarray(weights, dtype=self.bank.params.dtype), *batch)
            for batch in cached
        ]
        mse, terminal = jax.device_get(
            (
                sum(value[0] for value in values),
                sum(value[1] for value in values),
            )
        )
        scales = np.asarray(self.scales)
        return {
            "ensemble_component_rmse": np.sqrt(mse).tolist(),
            "ensemble_standardized_rmse": float(np.sqrt(np.mean(mse / scales**2))),
            "ensemble_position_rmse": float(np.sqrt(mse[:2].sum())),
            "ensemble_terminal_position_rmse": float(np.sqrt(terminal[:2].sum())),
        }

    def warmup(self, state):
        super().warmup(state)
        windows = RolloutBatch(
            np.asarray(state)[None],
            np.zeros((1, self.horizon, 2)),
            np.broadcast_to(state, (1, self.horizon, 7)),
            np.full((1, self.horizon), self.bank.timestep),
            np.full((1, self.horizon), self.bank.integrator_timestep),
            np.ones((1, self.horizon), dtype=bool),
        )
        for batch in self._cache(windows, np.ones(1)):
            jax.block_until_ready(
                blend_loss_and_grad(
                    jnp.asarray(self.weights, dtype=self.bank.params.dtype), *batch, self.scales
                )
            )
        # New-attempt diagnostics are one-step predictions, separate from the
        # horizon objective; warm that fixed-shape cache before publication too.
        self.score_warmup(state)

    def score_warmup(self, state):
        from sit_lmpc.data.dynamics import TransitionBatch

        self.score(
            TransitionBatch(
                np.asarray(state)[None],
                np.zeros((1, 2)),
                np.asarray(state)[None],
                np.asarray([self.bank.timestep]),
                np.asarray([self.bank.integrator_timestep]),
                np.zeros((1, 2), dtype=int),
            ),
            self.weights,
        )

    def score(self, transitions, weights):
        result = super().score(transitions, weights)
        if not len(transitions):
            return result
        windows = RolloutBatch(
            transitions.states,
            transitions.controls[:, None],
            transitions.next_states[:, None],
            transitions.intervals[:, None],
            transitions.substeps[:, None],
            np.ones((len(transitions), 1), dtype=bool),
        )
        try:
            result.update(
                self._metrics(
                    weights, self._cache(windows, np.full(len(windows), 1 / len(windows)))
                )
            )
        except FloatingPointError:
            result["ensemble_status"] = "nonfinite_prediction"
        result["weighted_component_rmse_kind"] = "expected_member_error_not_ensemble_mean"
        return result

    def _fit_mean(self, cached):
        """Regression reference, sharing the exact cache/objective with distribution fitting."""
        best_weights, best_loss = self.weights.copy(), np.inf

        def evaluate(candidate):
            nonlocal best_weights, best_loss
            w = jnp.asarray(candidate, dtype=self.bank.params.dtype)
            values = [blend_loss_and_grad(w, *batch, self.scales) for batch in cached]
            loss, gradient = jax.device_get(
                (
                    sum(value[0] for value in values),
                    sum(value[1] for value in values),
                )
            )
            loss = float(loss)
            if (
                np.isfinite(loss)
                and np.min(candidate) >= -1e-8
                and abs(candidate.sum() - 1) < 1e-8
                and loss < best_loss
            ):
                best_weights, best_loss = candidate.copy(), loss
            return loss, np.asarray(gradient, dtype=float)

        initial_loss, _ = evaluate(self.weights)
        if np.count_nonzero(self.prior) == 1:
            iterations, converged, message = 0, True, "single supported model"
        else:
            result = minimize(
                evaluate,
                self.weights,
                jac=True,
                method="SLSQP",
                bounds=[(0, 1) if prior > 0 else (0, 0) for prior in self.prior],
                constraints={
                    "type": "eq",
                    "fun": lambda w: w.sum() - 1,
                    "jac": lambda w: np.ones_like(w),
                },
                options={
                    "maxiter": self.config.fit_max_iterations,
                    "ftol": self.config.fit_tolerance,
                },
            )
            iterations, converged, message = (
                int(result.nit),
                bool(result.success),
                str(result.message),
            )
        best_weights = np.maximum(best_weights, 0)
        best_weights /= best_weights.sum()
        return best_weights, {
            "initial_loss": initial_loss,
            "loss": best_loss,
            "optimizer_iterations": iterations,
            "optimizer_converged": converged,
            "optimizer_message": message,
        }

    def fit(self, selection, generation):
        if not len(selection.transitions):
            return None, {"status": "no_valid_transitions"}
        if selection.windows is None:
            raise ValueError("ensemble_fit requires recorded rollout windows")
        if selection.windows.controls.shape[1] != self.horizon:
            raise ValueError("ensemble_fit window length must match its configured horizon")
        # Regression coefficients remain distinct from the legacy likelihood distribution.
        posterior, likelihood_metrics = super().fit(selection, generation)
        try:
            cached = self._cache(selection.windows, selection.weights)
        except FloatingPointError:
            return None, {"status": "nonfinite_model_rollout", "horizon": self.horizon}
        best_weights, regression = self._fit_mean(cached)
        best_loss = regression["loss"]
        metrics = {
            **self._metrics(best_weights, cached),
            **regression,
            "horizon": self.horizon,
            "windows": len(selection.windows),
            "posterior_status": likelihood_metrics["status"],
            "posterior_fit": likelihood_metrics,
            "weight_kind": "ensemble_coefficients",
        }
        if (
            not np.isfinite(best_loss)
            or metrics["ensemble_standardized_rmse"] > self.config.max_standardized_rmse
        ):
            return None, {**metrics, "status": "ensemble_mismatch"}
        published = jnp.asarray(best_weights, dtype=self.bank.params.dtype)
        self.weights = np.array(published, dtype=float, copy=True)
        self.weights /= self.weights.sum()
        return BeliefSnapshot(
            self.bank,
            published,
            generation,
            "ensemble_fit",
            None if posterior is None else posterior.weights,
        ), {**metrics, "status": "ready"}
