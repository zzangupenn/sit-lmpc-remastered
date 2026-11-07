"""Standalone BRO-compatible particles; actual BRO predictor wiring is separate."""

from dataclasses import dataclass
import math
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TrajectoryScenarios:
    states: jax.Array
    weights: jax.Array
    valid: jax.Array
    probability_error_bound: jax.Array | None = None
    verification_confidence: float | None = struct.field(pytree_node=False, default=None)


@dataclass(frozen=True)
class GaussianScenarioProvider:
    posterior: object
    scenario_count: int = 64
    action_scale: tuple[float, float] = (0.4, 3.5)
    controls_normalized: bool = True

    def __post_init__(self):
        if (
            type(self.scenario_count) is not int
            or self.scenario_count < 1
            or len(self.action_scale) != 2
            or min(self.action_scale) <= 0
        ):
            raise ValueError("positive scenario count and action scales required")
        if (
            self.controls_normalized
            and len(self.action_scale) != self.posterior.model.domain.control_dimension
        ):
            raise ValueError(
                "action_scale must match the posterior domain; use native controls for HalfCheetah"
            )

    @property
    def metadata(self):
        return {
            "method": "gaussian_last_layer_mc",
            "sample_count": self.scenario_count,
            "probability_source": "gaussian_posterior"
            if self.posterior.seen_ids
            else "gaussian_population_prior",
            "belief_generation": self.posterior.generation,
            "population_generation": self.posterior.model.generation,
            "model_fingerprint": self.posterior.model.fingerprint,
            "confidence": None,
            "uncertainty": "epistemic only; execution noise excluded",
        }

    def rollout(self, state, controls, *, key, scan=True):
        # Always use the compiled scan; the flag does not change trajectory semantics.
        del scan
        if key is None:
            raise ValueError("a caller-owned random key is required for sampled dynamics")
        functions = self.posterior.sample_functions(key, self.scenario_count)
        controls = jnp.asarray(controls, dtype=functions.coefficients.dtype)
        if self.controls_normalized:
            controls = controls * jnp.asarray(self.action_scale, dtype=controls.dtype)
        states = functions.rollout(state, controls)
        finite = jnp.all(jnp.isfinite(states), axis=-1)
        valid = jnp.cumprod(finite, axis=-1).astype(bool)
        return TrajectoryScenarios(states, functions.weights, valid, None)

    def verification_rollout(
        self, state, fixed_controls, *, planning_key, validation_key, confidence=0.95
    ):
        """IID Hoeffding envelope for a finite set fixed before validation.

        Covers two geometric events across all candidates by a union bound.
        Controls must not be optimized using this validation batch afterward.
        Different keys alone cannot establish independence from arbitrary prior use;
        caller must reserve validation_key exclusively for this validation.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in (0,1)")
        if np.array_equal(jax.random.key_data(planning_key), jax.random.key_data(validation_key)):
            raise ValueError("validation must not reuse the planning key")
        candidates = len(fixed_controls) if np.ndim(fixed_controls) == 3 else 1
        scenarios = self.rollout(state, fixed_controls, key=validation_key)
        error = min(
            1.0, math.sqrt(math.log(2 * candidates / (1 - confidence)) / (2 * self.scenario_count))
        )
        return scenarios.replace(
            probability_error_bound=jnp.asarray(error, dtype=scenarios.states.dtype),
            verification_confidence=confidence,
        )


@dataclass(frozen=True)
class MeanPredictor:
    """Point prediction convenience, not the existing BRO STModel dispatch type."""

    posterior: object
    action_scale: tuple[float, float] = (0.4, 3.5)

    def step(self, state, normalized_action):
        return self.posterior.predict(
            state, jnp.asarray(normalized_action) * jnp.asarray(self.action_scale)
        ).mean

    def rollout(self, state, normalized_controls):
        def advance(x, u):
            target = self.step(x, u)
            return target, target

        dtype = self.posterior.model.normalizer.input_mean.dtype
        return jax.lax.scan(
            advance, jnp.asarray(state, dtype=dtype), jnp.asarray(normalized_controls, dtype=dtype)
        )[1]
