"""Diagnostic-only model comparisons with bounded, aligned measured history."""

from collections import deque
from dataclasses import dataclass
import time
import jax
import jax.numpy as jnp
import numpy as np

from sit_lmpc.dynamics.bank import STModelBank, bank_step, bank_rollout
from sit_lmpc.dynamics.st import PARAMETER_FIELDS
from sit_lmpc.dynamics.ensemble import (
    STEnsemble, predictor_rollout as rollout, predictor_friction, member_rollouts,
)


@dataclass(frozen=True)
class Comparison:
    kind: str
    step: int
    actions: np.ndarray
    candidates: np.ndarray
    belief: np.ndarray
    true_reference: np.ndarray
    actual: np.ndarray
    frictions: tuple[float, ...]
    belief_friction: float | None
    true_friction: float
    timestep: float
    evaluation_seconds: float
    model_weights: np.ndarray | None = None
    belief_generation: int = 0
    belief_mode: str = "single"
    weight_kind: str = "model_probabilities"


class ModelComparisons:
    def __init__(self, belief, true_reference, frictions, horizon, scan=True):
        self.bnn = hasattr(belief, "functions")
        self.bank = None if self.bnn else belief.bank if isinstance(belief, STEnsemble) else STModelBank.from_model(belief, frictions)
        self.belief, self.true_reference = belief, true_reference
        self.frictions, self.horizon, self.scan = tuple(frictions), horizon, scan
        if self.bnn:
            self.frictions = (None,) * len(belief.weights)
        self.states = deque(maxlen=horizon + 1)
        self.actions = deque(maxlen=horizon)
        self.warmup_seconds = None
        self.steps = 0
        mu = PARAMETER_FIELDS.index("mu")
        self.belief_friction = predictor_friction(belief)
        self.true_friction = float(true_reference.params[mu])
        self.model_weights, self.belief_generation = None, 0
        self.weight_kind = getattr(belief, "weight_kind", "model_probabilities")
        if self.bnn:
            self.set_belief(belief)

    def set_belief(self, model, snapshot=None):
        """Ground truth remains diagnostic-only; model support comes from learning."""
        self.belief = model
        self.belief_friction = predictor_friction(model)
        if self.bnn:
            # Reuse this control update's frozen planning batch, not fresh draws
            # or the independent chance-verification batch. Do not consume RNG.
            self.frictions = (None,) * len(model.weights)
            self.model_weights = np.asarray(model.weights).copy()
            self.belief_generation = int(model.belief_generation)
            self.weight_kind = "gaussian_function_samples"
            return
        self.bank = snapshot.bank
        self.model_weights = np.asarray(snapshot.weights).copy()
        self.belief_generation = snapshot.generation
        self.weight_kind = snapshot.weight_kind

    def warmup(self, state):
        state = jnp.asarray(state, dtype=self.belief.params.dtype)
        actions = jnp.zeros((self.horizon, 2), dtype=self.belief.params.dtype)
        started = time.perf_counter()
        jax.block_until_ready(
            (
                self.belief.step_members(state, actions[0]) if self.bnn else bank_step(self.bank, state, actions[0]),
                member_rollouts(self.belief, state, actions, self.scan) if self.bnn else bank_rollout(self.bank, state, actions, self.scan),
                rollout(self.belief, state, actions, self.scan),
                rollout(self.true_reference, state, actions, self.scan),
            )
        )
        self.warmup_seconds = time.perf_counter() - started

    def observe(self, state, applied_action, next_state):
        self.steps += 1
        if not self.states:
            self.states.append(np.asarray(state).copy())
        self.actions.append(np.asarray(applied_action).copy())
        self.states.append(np.asarray(next_state).copy())

    def reset_history(self):
        """Never replay a reset teleport as a physical transition."""
        self.states.clear()
        self.actions.clear()

    def _evaluate(self, state, actions, actual, kind, step):
        if self.bnn:
            # Refresh labels/counts for the latest planner batch only on demand;
            # ordinary non-debug control steps need no extra device transfers.
            self.set_belief(self.belief)
        state = np.asarray(state, dtype=self.belief.params.dtype)
        actions = np.asarray(actions, dtype=self.belief.params.dtype)
        actual = np.asarray(actual, dtype=self.belief.params.dtype)
        count = len(actions)
        if state.shape != (7,) or actions.shape != (count, 2) or not 0 < count <= self.horizon:
            raise ValueError("comparison expects a state and a nonempty bounded action horizon")
        if actual.shape != ((count + 1, 7) if kind == "recorded" else (1, 7)):
            raise ValueError("comparison actual states do not match the recorded/planned horizon")
        if not all(np.isfinite(a).all() for a in (state, actions, actual)):
            raise FloatingPointError("nonfinite comparison input")
        # A partial history uses zero padding, not a new compiled horizon shape.
        padded = np.zeros((self.horizon, 2), dtype=actions.dtype)
        padded[:count] = actions
        started = time.perf_counter()
        device_state, device_actions = jnp.asarray(state), jnp.asarray(padded)
        candidates, belief, truth = jax.device_get(
            (
                member_rollouts(self.belief, device_state, device_actions, self.scan) if self.bnn else bank_rollout(self.bank, device_state, device_actions, self.scan),
                rollout(self.belief, device_state, device_actions, self.scan),
                rollout(self.true_reference, device_state, device_actions, self.scan),
            )
        )
        elapsed = time.perf_counter() - started
        candidates = np.concatenate(
            (np.broadcast_to(state, (len(self.frictions), 1, 7)), candidates[:, :count]), axis=1
        )
        belief = np.vstack((state, belief[:count]))
        truth = np.vstack((state, truth[:count]))
        if not all(np.isfinite(a).all() for a in (candidates, belief, truth)):
            raise FloatingPointError("nonfinite model comparison rollout")
        return Comparison(
            kind,
            step,
            actions.copy(),
            candidates,
            belief,
            truth,
            actual.copy(),
            self.frictions,
            self.belief_friction,
            self.true_friction,
            self.belief.timestep,
            elapsed,
            None if self.model_weights is None else self.model_weights.copy(),
            self.belief_generation,
            "gaussian_bnn" if self.bnn else "ensemble" if isinstance(self.belief, STEnsemble) else "single",
            self.weight_kind,
        )

    def planned(self, state, actions, step):
        # The future actual Gym trajectory is unavailable before applying controls.
        return self._evaluate(
            state, np.asarray(jax.device_get(actions)), np.asarray(state)[None], "planned", step
        )

    def recorded(self, step):
        if not self.actions:
            return None
        actual = np.asarray(self.states)
        return self._evaluate(actual[0], np.asarray(self.actions), actual, "recorded", step)
