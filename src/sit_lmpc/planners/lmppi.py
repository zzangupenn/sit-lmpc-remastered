"""Learning MPPI: fixed analytical ST dynamics and a learned terminal value."""

from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np
from sit_lmpc.dynamics.ensemble import (
    STEnsemble, predictor_rollout as rollout, rollout_prediction,
)
from sit_lmpc.training.value import predict_value
from sit_lmpc.training.risk import predict_risk, RiskSnapshot
from sit_lmpc.training.features import inference_snapshot
from .common import PlannerResult, returns, unit_range, weights, to_frenet
from .safe_set import build_safe_geometry, hull_distance, PrecompiledHullEvaluator
from .implicit_hull import ImplicitHullEvaluator
from sit_lmpc.training.safe_set_density import (
    DensitySnapshot, density_penalty, density_scores, density_violation, relaxation_cutoff,
    sample_normalized_density, density_method_metadata,
)
from .probabilistic import (
    SafetyStop, chance_probabilities, distribution_availability, lambda_table,
    select_constrained_candidate, configured_candidate_counts,
)


@jax.jit
def terminal_summary(states):
    valid = jnp.all(jnp.isfinite(states), axis=(1, 2))
    return (jnp.min(jnp.where(valid, states[:, -1, 0], jnp.inf)),
            jnp.max(jnp.where(valid, states[:, -1, 0], -jnp.inf)), jnp.any(valid))


@partial(jax.jit, static_argnames=("config",))
def sample_actions(previous, key, config):
    if previous.shape not in ((config.horizon, 2), (config.horizon + 1, 2)):
        raise ValueError("action sequence must contain H or H+1 controls")
    std = jnp.asarray(config.sample_std, dtype=previous.dtype)
    prefix = previous[: config.horizon]
    da = (
        jax.random.truncated_normal(
            key,
            (-1 - prefix) / std,
            (1 - prefix) / std,
            shape=(config.samples, config.horizon, 2),
        )
        * std
    )
    if len(previous) == config.horizon + 1:
        # Keep the original in-horizon draws identical for a given seed.
        tail = (
            jax.random.truncated_normal(
                jax.random.fold_in(key, 1),
                (-1 - previous[-1:]) / std,
                (1 - previous[-1:]) / std,
                shape=(config.samples, 1, 2),
            )
            * std
        )
        da = jnp.concatenate((da, tail), axis=1)
    actions = jnp.clip(previous[None] + da, -1, 1)
    return da, actions


@partial(jax.jit, static_argnames=("config",))
def sample_rollouts(model, state, previous, key, config):
    da, actions = sample_actions(previous, key, config)
    states = jax.vmap(lambda a: rollout(model, state, a, config.scan_rollout))(
        actions[:, : config.horizon]
    )
    return states, da, actions


@partial(jax.jit, static_argnames=("config",))
def sample_predictions(model, state, previous, key, config):
    """Share independent member paths with probability evaluators when possible."""
    da, actions = sample_actions(previous, key, config)
    prediction, members = candidate_predictions(model, state, actions, config)
    return prediction, members, da, actions


def sample_validity(previous, da, states, values, hull_distances):
    return (jnp.all(jnp.isfinite(states), axis=(1, 2))
            & jnp.all(jnp.isfinite(values.reshape(len(states), -1)), axis=1)
            & jnp.isfinite(hull_distances)
            & jnp.all(jnp.isfinite(da), axis=(1, 2))
            & jnp.all(jnp.isfinite(previous)))


def _reward_terms(previous, da, states, values, hull_distances, lambdas, config, fixed_density_cost=False):
    valid = sample_validity(previous, da, states, values, hull_distances)
    terminal = jnp.zeros(config.horizon).at[-1].set(1)
    boundary = jnp.abs(jnp.where(valid[:, None], states[:, :, 1], 0.))
    r_boundary = unit_range(jnp.where(boundary > config.boundary_limit, boundary, 0), valid[:, None])
    r_value = values.reshape(config.samples, -1)
    if config.mean_value:
        r_value = r_value.mean(axis=1, keepdims=True)
    r_value = unit_range(r_value, valid[:, None]) * terminal
    distance = jnp.where(
        valid & (hull_distances > config.safe_set_threshold),
        hull_distances - config.safe_set_threshold,
        0,
    )
    # Sigma NLL penalties already have fixed 0/1 boundaries. Batch
    # normalization would turn an all-saturated batch into zero penalty.
    hull_cost = distance if fixed_density_cost else unit_range(jnp.minimum(distance, config.safe_set_reward_limit), valid)
    r_hull = hull_cost[:, None] * terminal
    horizon_actions = (da + previous)[:, : config.horizon]
    action_changes = jnp.linalg.norm(horizon_actions[:, 1:] - horizon_actions[:, :-1], axis=-1)
    r_action = jnp.pad(action_changes, ((0, 0), (1, 0))) * config.action_rate_penalty
    reward = (
        r_value[None]
        - lambdas[:, 0, None, None] * r_hull[None]
        - lambdas[:, 1, None, None] * r_boundary[None]
        - r_action[None]
    )
    reward = jnp.where(valid[None, :, None], reward, -jnp.inf)
    return {"reward": reward, "value": r_value, "hull": r_hull, "boundary": r_boundary, "valid": valid}


debug_reward_terms = jax.jit(_reward_terms, static_argnames=("config", "fixed_density_cost"))


@partial(jax.jit, static_argnames=("config", "fixed_density_cost"))
def propose(previous, da, states, values, hull_distances, lambdas, config, fixed_density_cost=False):
    terms = _reward_terms(previous, da, states, values, hull_distances, lambdas, config, fixed_density_cost)
    w = weights(returns(terms["reward"]), config.temperature, config.damping, valid=terms["valid"][None, :, None])
    if len(previous) == config.horizon + 1:
        # u_H uses the terminal weights from the SAME sampled sequences.
        w = jnp.concatenate((w, w[:, :, -1:]), axis=2)
    noise = jnp.where(terms["valid"][:, None, None], da, 0.)
    return jnp.clip(previous + jnp.sum(w[:, :, :, None] * noise[None], axis=1), -1, 1)


def normalized_risk_cost(risk_probabilities, mode):
    """Normalize sampled risks; retain each step in all-state mode."""
    if mode == "all":
        return unit_range(risk_probabilities)
    if mode == "terminal":
        terminal_risk = unit_range(risk_probabilities[:, -1])
        return jnp.zeros_like(risk_probabilities).at[:, -1].set(terminal_risk)
    raise ValueError("risk mode must be all or terminal")


@partial(jax.jit, static_argnames=("config", "mode", "fixed_density_cost"))
def propose_with_risk(
    previous, da, states, values, hull_distances, risk_probabilities, lambdas, config, mode,
    fixed_density_cost=False,
):
    """Penalize each selected risk value; retain the separate cutoff gate."""
    terms = _reward_terms(previous, da, states, values, hull_distances, lambdas, config, fixed_density_cost)
    risk_cost = normalized_risk_cost(risk_probabilities, mode)
    reward = terms["reward"] - lambdas[:, 2, None, None] * risk_cost[None]
    w = weights(returns(reward), config.temperature, config.damping, valid=terms["valid"][None, :, None])
    if len(previous) == config.horizon + 1:
        w = jnp.concatenate((w, w[:, :, -1:]), axis=2)
    noise = jnp.where(terms["valid"][:, None, None], da, 0.)
    return jnp.clip(previous + jnp.sum(w[:, :, :, None] * noise[None], axis=1), -1, 1)


@partial(jax.jit, static_argnames=("config", "mode", "fixed_density_cost"))
def propose_with_constraints(previous, da, states, values, hull_distances,
                             chance, risk, lambdas, config, mode, fixed_density_cost=False):
    """Optional arguments are absent, not zero arrays or unused lambda columns."""
    terms = _reward_terms(previous, da, states, values, hull_distances, lambdas, config, fixed_density_cost)
    reward = terms["reward"]
    column = 2
    if chance is not None:
        terminal = jnp.zeros(config.horizon, dtype=states.dtype).at[-1].set(1)
        # Path-level probabilities already lie in [0,1]; count each event once.
        reward -= lambdas[:, column, None, None] * chance["boundary"][None, :, None] * terminal
        reward -= lambdas[:, column + 1, None, None] * chance["terminal_hull"][None, :, None] * terminal
        column += 2
    if risk is not None:
        reward -= lambdas[:, column, None, None] * normalized_risk_cost(risk, mode)[None]
    w = weights(returns(reward), config.temperature, config.damping, valid=terms["valid"][None, :, None])
    if len(previous) == config.horizon + 1:
        w = jnp.concatenate((w, w[:, :, -1:]), axis=2)
    noise = jnp.where(terms["valid"][:, None, None], da, 0.)
    return jnp.clip(previous + jnp.sum(w[:, :, :, None] * noise[None], axis=1), -1, 1)


@partial(jax.jit, static_argnames=("config",))
def candidate_rollouts(model, state, candidates, config):
    return jax.vmap(lambda a: rollout(model, state, a, config.scan_rollout))(
        candidates[:, : config.horizon]
    )


@partial(jax.jit, static_argnames=("config",))
def candidate_predictions(model, state, candidates, config):
    return jax.vmap(lambda a: rollout_prediction(model, state, a, config.scan_rollout))(
        candidates[:, : config.horizon]
    )


@jax.jit
def endpoint_query_inputs(states, actions, action_scale):
    """Shared terminal (x_H, u_H) pairing for action-conditioned value/risk."""
    if actions.shape != (len(states), states.shape[1] + 1, 2):
        raise ValueError("endpoint queries require H+1 actions for H predicted states")
    return states[:, -1], actions[:, -1] * action_scale


@partial(jax.jit, static_argnames=("mean_value",))
def value_query_inputs(states, initial_frenet, actions, action_scale, mean_value):
    """Pair rollout states with NEXT controls, never with their incoming controls."""
    if mean_value:
        initial = jnp.broadcast_to(initial_frenet, (len(states), 1, 7))
        query = jnp.concatenate((initial, states[:, :-1]), axis=1)
        controls = actions[:, : states.shape[1]] * action_scale
    else:
        query, controls = endpoint_query_inputs(states, actions, action_scale)
    return query.reshape(-1, 7), controls.reshape(-1, 2)


def sampled_risk_diagnostic(snapshot, states, sampled_actions, action_scale):
    """Score x_1..x_H, using u_1..u_H for action-conditioned risk."""
    states = jnp.asarray(states)
    count, horizon = states.shape[:2]
    probabilities = rollout_risk_probabilities(snapshot, states, sampled_actions, action_scale)
    return jax.device_get(
        {
            "probabilities": probabilities,
            "frenet": states[:, :, :2],
            "generation": snapshot.generation,
            "input_mode": snapshot.input_mode,
        }
    )


def _rollout_risk_probabilities(snapshot, states, actions, action_scale):
    """Predict risk at x_1..x_H, paired with u_1..u_H when action-conditioned."""
    count, horizon = states.shape[:2]
    risk_actions = None
    if snapshot.input_mode == "state_action":
        if actions is None or actions.shape != (count, horizon + 1, 2):
            raise ValueError("action-conditioned risk needs H+1 sampled controls")
        risk_actions = (jnp.asarray(actions)[:, 1:] * action_scale).reshape(-1, 2)
    probabilities = predict_risk(snapshot, states.reshape(-1, 7), risk_actions).reshape(
        count, horizon
    )
    return probabilities


_compiled_rollout_risk = jax.jit(_rollout_risk_probabilities)


def rollout_risk_probabilities(snapshot, states, actions, action_scale):
    # Retain duck-typed diagnostic/testing snapshots without weakening the real path.
    function = _compiled_rollout_risk if isinstance(snapshot, RiskSnapshot) else _rollout_risk_probabilities
    return function(snapshot, states, actions, action_scale)


def _distribution_risk_probabilities(snapshot, scenarios, frenet, actions, action_scale):
    """Average per-step learned risks, not coordinates or time; no density twice."""
    count, members, horizon = frenet.shape[:3]
    member_actions = jnp.broadcast_to(actions[:, None], (count, members, *actions.shape[1:]))
    probabilities = _rollout_risk_probabilities(
        snapshot, frenet.reshape(count * members, horizon, 7),
        member_actions.reshape(count * members, *actions.shape[1:]), action_scale,
    ).reshape(count, members, horizon)
    probabilities = jnp.where(scenarios.valid, probabilities, 1.0)
    return jnp.sum(jnp.where(scenarios.weights[None, :, None] > 0,
                             scenarios.weights[None, :, None] * probabilities, 0), axis=1)


_compiled_distribution_risk = jax.jit(_distribution_risk_probabilities)


def distribution_risk_probabilities(snapshot, scenarios, frenet, actions, action_scale):
    function = _compiled_distribution_risk if isinstance(snapshot, RiskSnapshot) else _distribution_risk_probabilities
    return function(snapshot, scenarios, frenet, actions, action_scale)


def candidate_scores(frenet, hull_distances, boundary_limit, relaxation, terminal_values=None):
    boundary = jnp.abs(frenet[:, :, 1])
    boundary_sum = jnp.where(boundary > boundary_limit, boundary, 0).sum(axis=1)
    violation = jnp.where(hull_distances > relaxation, hull_distances, 0)
    score = boundary_sum * 1000 + violation
    valid = (jnp.all(jnp.isfinite(frenet), axis=(1, 2))
             & jnp.isfinite(hull_distances) & jnp.isfinite(score))
    if terminal_values is not None:
        valid &= jnp.isfinite(terminal_values)
    return jnp.where(valid, score, jnp.inf), violation, boundary_sum, valid


@partial(jax.jit, static_argnames=("boundary_limit", "relaxation"))
def select_candidate(frenet, hull_distances, boundary_limit, relaxation, terminal_values=None):
    """Least finite violation, then first in ascending lambda-norm order."""
    score, violation, boundary_sum, _ = candidate_scores(
        frenet, hull_distances, boundary_limit, relaxation, terminal_values
    )
    index = jnp.argmin(score)
    return index, violation[index], boundary_sum[index]


@partial(jax.jit, static_argnames=("mode", "boundary_limit", "relaxation"))
def select_risk_candidate(
    frenet, hull_distances, risk_probabilities, cutoff, mode, boundary_limit, relaxation,
    terminal_values=None,
):
    """Gate risk, then use SIT's least-violation/first-lambda rule, even for Q values."""
    score, violation, boundary_sum, valid = candidate_scores(
        frenet, hull_distances, boundary_limit, relaxation, terminal_values
    )
    risk_score = jnp.max(risk_probabilities, axis=1) if mode == "all" else risk_probabilities[:, -1]
    feasible = valid & jnp.isfinite(risk_score) & (risk_score <= cutoff)
    gated_score = jnp.where(feasible, score, jnp.inf)
    feasible_index = jnp.argmin(gated_score)
    baseline_index = jnp.argmin(score)
    has_feasible = jnp.any(feasible)
    index = jnp.where(has_feasible, feasible_index, baseline_index)
    return index, violation[index], boundary_sum[index], risk_score[index], feasible.sum(), ~has_feasible


@partial(jax.jit, static_argnames=("config",))
def finish_candidates(frenet, distances, chance, risk, cutoff, terminal_values,
                      candidates, trajectories, members, probabilities, lambdas, config):
    """Gate and gather on device; diagnostics do not trigger extra scalar fences."""
    finite_controls = jnp.all(jnp.isfinite(candidates), axis=(1, 2))
    finite_trajectories = jnp.all(jnp.isfinite(trajectories), axis=(1, 2))
    distances = jnp.where(finite_controls & finite_trajectories, distances, jnp.inf)
    _, _, _, finite = candidate_scores(frenet, distances, config.boundary_limit,
                                        config.safe_set_threshold, terminal_values)
    feasible_count = fallback = chosen_risk = None
    if chance is not None:
        index, hull, boundary, feasible_count, fallback, risk_scores = select_constrained_candidate(
            frenet, distances, chance, risk, cutoff,
            config.probabilistic_safety.boundary_threshold,
            config.probabilistic_safety.terminal_hull_threshold,
            config.risk_constraint.mode, config.boundary_limit,
            config.safe_set_threshold, terminal_values,
        )
        chosen_risk = risk_scores[index] if risk is not None else None
    elif risk is not None:
        index, hull, boundary, chosen_risk, feasible_count, fallback = select_risk_candidate(
            frenet, distances, risk, cutoff, config.risk_constraint.mode,
            config.boundary_limit, config.safe_set_threshold, terminal_values,
        )
    else:
        index, hull, boundary = select_candidate(
            frenet, distances, config.boundary_limit, config.safe_set_threshold, terminal_values,
        )
    chosen = {name: values[index] for name, values in chance.items()} if chance is not None else None
    chance_count = None if chance is None else jnp.sum(
        chance["available"]
        & (chance["boundary_upper"] <= config.probabilistic_safety.boundary_threshold)
        & (chance["terminal_hull_upper"] <= config.probabilistic_safety.terminal_hull_threshold)
    )
    risk_count = None if risk is None else jnp.sum(
        (jnp.max(risk, axis=1) if config.risk_constraint.mode == "all" else risk[:, -1]) <= cutoff
    )
    return {
        "index": index, "hull": hull, "boundary": boundary, "actions": candidates[index],
        "finite_candidate_count": jnp.sum(finite),
        "trajectory": trajectories[index], "lambdas": lambdas[index],
        "terminal_value": None if terminal_values is None else terminal_values[index],
        "members": None if members is None else members[index], "probabilities": probabilities,
        "chosen_chance": chosen, "chosen_risk": chosen_risk,
        "feasible_count": feasible_count, "fallback": fallback,
        "chance_count": chance_count, "risk_count": risk_count,
        "minimum_boundary": None if chance is None else jnp.min(chance["boundary"]),
        "minimum_hull": None if chance is None else jnp.min(chance["terminal_hull"]),
    }


class LMPPI:
    def __init__(
        self,
        config,
        model,
        track,
        seed=0,
        value_samples=20,
        randomness=None,
        value_input_mode="state",
        risk_input_mode=None,
        scenario_provider=None,
        emit=lambda *args, **kwargs: None,
    ):
        self.config, self.model, self.track = config, model, track
        self.value_samples = value_samples
        self.key = jax.random.key(seed)
        self.randomness = randomness
        self.scenario_provider, self.emit = scenario_provider, emit
        self.value_input_mode, self.risk_input_mode = value_input_mode, risk_input_mode
        self.endpoint_action = value_input_mode == "state_action" and not config.mean_value
        self.actions = jnp.zeros(
            (config.horizon + int(self.endpoint_action), 2), dtype=model.params.dtype
        )
        grid = np.arange(config.lambda_count) * config.lambda_max / config.lambda_count
        lambdas = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
        lambdas = lambdas[np.argsort(np.linalg.norm(lambdas, axis=1))]
        self.lambdas = jnp.asarray(lambdas, dtype=model.params.dtype)
        self.risk_lambdas = None
        self._lambda_tables = {(False, False): self.lambdas}
        self._warmed_terms = set()
        self._risk_warmed = False
        self._window_prediction = None
        self.density_snapshot = None
        self.density_calibration = None
        self._density_metadata_identity, self._density_metadata = None, {}
        self.hull_evaluator = (
            ImplicitHullEvaluator(config, len(lambdas), model.params.dtype)
            if config.safe_set_query == "implicit" else
            PrecompiledHullEvaluator(
                config.samples,
                len(lambdas),
                config.hull_bucket,
                config.hull_precompile_limit,
                model.params.dtype,
            )
            if config.hull_precompile
            else None
        )

    def precompile_hulls(self, emit=lambda *args, **kwargs: None, *, force=False):
        if not force and self.config.safe_set_query == "nf_density":
            return
        if self.hull_evaluator is not None and self.config.hull_precompile:
            self.hull_evaluator.precompile(emit)
            provider = self.scenario_provider
            if self.config.probabilistic_safety.enabled and provider is not None:
                members = provider.scenario_count
                counts = {self.config.samples * members}
                for count in configured_candidate_counts(self.config):
                    counts.add(count * members)
                    if getattr(provider, "uses_independent_verification", False):
                        counts.add(count * provider.verification_particles)
                self.hull_evaluator.precompile_many(counts, emit)
        # Optional term tables/kernels are warmed only once their inputs exist.

    def active_lambdas(self, chance_active, risk_active, *, force_hull=False):
        key = chance_active, risk_active
        if key not in self._lambda_tables:
            self._lambda_tables[key] = lambda_table(self.lambdas, self.config, *key)
        table = self._lambda_tables[key]
        if key == (False, True):
            self.risk_lambdas = table
        if (self.hull_evaluator is not None and self.config.hull_precompile
                and (force_hull or not (self.config.safe_set_query == "nf_density" and self.density_snapshot is not None))):
            self.hull_evaluator.ensure_batch(len(table), self.emit)
        fixed_density_cost = (not force_hull and self.density_snapshot is not None
                              and self.density_snapshot.fixed_penalty)
        proposal_options = {"fixed_density_cost": True} if fixed_density_cost else {}
        warm_key = (*key, len(self.actions), fixed_density_cost)
        if warm_key not in self._warmed_terms:
            dtype = self.model.params.dtype
            sample_count, horizon = self.config.samples, self.config.horizon
            candidate_count = len(table)
            previous = jnp.zeros_like(self.actions)
            da = jnp.zeros((sample_count, len(previous), 2), dtype=dtype)
            states = jnp.zeros((sample_count, horizon, 7), dtype=dtype)
            value_count = sample_count if self.value_input_mode == "state_action" and not self.config.mean_value else sample_count * horizon
            values = jnp.zeros(value_count, dtype=dtype)
            risk = jnp.zeros((sample_count, horizon), dtype=dtype) if risk_active else None
            chance = {name: jnp.zeros(sample_count, dtype=dtype)
                      for name in ("boundary", "terminal_hull")} if chance_active else None
            if chance_active:
                proposed = propose_with_constraints(
                    previous, da, states, values, jnp.zeros(sample_count, dtype=dtype),
                    chance, risk, table, self.config, self.config.risk_constraint.mode,
                    **proposal_options,
                )
            elif risk_active:
                proposed = propose_with_risk(
                    previous, da, states, values, jnp.zeros(sample_count, dtype=dtype),
                    risk, table, self.config, self.config.risk_constraint.mode,
                    **proposal_options,
                )
            else:
                proposed = propose(previous, da, states, values,
                                   jnp.zeros(sample_count, dtype=dtype), table, self.config, **proposal_options)
            proposed.block_until_ready()
            state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=dtype)
            jax.block_until_ready(candidate_predictions(self.model, state, proposed, self.config))
            candidate_states = jnp.zeros((candidate_count, horizon, 7), dtype=dtype)
            candidate_distances = jnp.zeros(candidate_count, dtype=dtype)
            candidate_risk = jnp.zeros((candidate_count, horizon), dtype=dtype) if risk_active else None
            if chance_active:
                candidate_chance = {
                    name: jnp.zeros(candidate_count, dtype=dtype)
                    for name in ("boundary", "terminal_hull", "boundary_upper", "terminal_hull_upper")
                }
                candidate_chance["available"] = jnp.ones(candidate_count, dtype=bool)
                checked = select_constrained_candidate(
                    candidate_states, candidate_distances, candidate_chance, candidate_risk, 0.5,
                    self.config.probabilistic_safety.boundary_threshold,
                    self.config.probabilistic_safety.terminal_hull_threshold,
                    self.config.risk_constraint.mode, self.config.boundary_limit,
                    self.config.safe_set_threshold,
                )
            elif risk_active:
                checked = select_risk_candidate(
                    candidate_states, candidate_distances, candidate_risk, 0.5,
                    self.config.risk_constraint.mode, self.config.boundary_limit,
                    self.config.safe_set_threshold,
                )
            else:
                checked = select_candidate(candidate_states, candidate_distances,
                                           self.config.boundary_limit,
                                           self.config.safe_set_threshold)
            jax.block_until_ready(checked)
            jax.block_until_ready(finish_candidates(
                candidate_states, candidate_distances,
                candidate_chance if chance_active else None, candidate_risk,
                0.5 if risk_active else None,
                jnp.zeros(candidate_count, dtype=dtype)
                if self.value_input_mode == "state_action" and not self.config.mean_value else None,
                proposed, candidate_states,
                jnp.zeros((candidate_count, len(self.model.weights), horizon, 7), dtype=dtype)
                if hasattr(self.model, "weights") else None,
                self.model.weights if hasattr(self.model, "weights") else None,
                table, self.config,
            ))
            self._warmed_terms.add(warm_key)
            self._risk_warmed |= risk_active
            self.emit(
                "probabilistic_safety_warmed" if chance_active else
                "risk_constraint_warmed" if risk_active else "controller_kernels_warmed",
                candidates=candidate_count,
                mode=self.config.risk_constraint.mode,
            )
        return table

    def evaluate_hull(self, states, hull):
        if isinstance(hull, DensitySnapshot):
            return density_penalty(hull, states)
        if self.config.safe_set_query == "nf_density":
            raise ValueError("nf_density cannot score a geometric safe set")
        if self.hull_evaluator is None:
            result = hull_distance(states, hull)
        else:
            # The runner calls this before driving; direct users can warm explicitly.
            self.precompile_hulls(force=True)
            result = self.hull_evaluator.evaluate(states, hull)
        return result

    def evaluate_safe_set_check(self, states, hull):
        if isinstance(hull, DensitySnapshot):
            if hull.relaxation_cutoff is not None:
                return density_violation(hull, states)
            raise ValueError("independent NF relaxation requires a bound but its calibrated bound is missing")
        return self.evaluate_hull(states, hull)

    def scenarios(self, provider, state, actions, key, indices, current_s, members=None, anchor=None, supplied=None):
        if supplied is not None:
            scenarios = supplied
        elif members is not None and getattr(provider, "can_reuse", lambda model: False)(self.model):
            scenarios = provider.from_members(members)
        else:
            scenarios = provider.rollout(
                state, actions[:, :self.config.horizon], key=key, scan=self.config.scan_rollout
            )
        count = len(scenarios.weights) if supplied is not None else provider.scenario_count
        expected = (len(actions), count, self.config.horizon, 7)
        if (scenarios.states.shape != expected or scenarios.valid.shape != expected[:-1]
                or scenarios.weights.shape != (count,)
                or scenarios.valid.dtype != jnp.bool_):
            raise ValueError("scenario provider returned incompatible trajectory/weight/mask shapes")
        if scenarios.probability_error_bound is not None:
            error_bound = jnp.asarray(scenarios.probability_error_bound, dtype=scenarios.states.dtype)
            if error_bound.shape != ():
                raise ValueError("scenario probability error bound must be a scalar")
            scenarios = scenarios.replace(probability_error_bound=error_bound)
        finite = jnp.all(jnp.isfinite(scenarios.states), axis=-1)
        # Mask placeholders before track lookup; a NaN may otherwise poison all
        # progress coordinates, including independent valid scenarios.
        sanitized = jnp.where((finite & scenarios.valid)[..., None], scenarios.states, state)
        frenet = to_frenet(self.track, sanitized, indices, current_s,
                           self.config.projection_block_size)
        if anchor is not None and hasattr(self.track, "s_frame_max"):
            length = self.track.s_frame_max
            progress = anchor[:, None, :, 0]
            delta = frenet[..., 0] - progress
            # All branches use the same lifted lap frame as the selected hull.
            frenet = frenet.at[..., 0].set(progress + (delta + length / 2) % length - length / 2)
        frenet = jnp.where(finite[..., None], frenet, jnp.nan)
        return scenarios, frenet

    def chance(self, scenarios, frenet, hull, *, empirical=False):
        # Reuse the original bounded N-query hull executables for each member;
        # no new hull-facet or N*M executables for every belief update.
        if self.config.safe_set_query == "nf_density" and not isinstance(hull, DensitySnapshot):
            raise ValueError("nf_density cannot score a geometric safe set")
        if isinstance(hull, DensitySnapshot):
            queries = frenet[:, :, -1, :5].reshape(-1, 5)
            distances = self.evaluate_safe_set_check(queries, hull).reshape(frenet.shape[:2])
        elif self.hull_evaluator is not None:
            queries = frenet[:, :, -1, :5].reshape(-1, 5)
            distances = self.hull_evaluator.evaluate_many(queries, hull).reshape(frenet.shape[:2])
        else:
            distances = jnp.stack([
                self.evaluate_hull(frenet[:, member, -1, :5], hull)
                for member in range(frenet.shape[1])
            ], axis=1)
        error = scenarios.probability_error_bound
        if empirical:
            # Proposals or an explicitly selected empirical planning-sample gate.
            # No added margin here does not mean zero sampling error: empirical
            # final diagnostics report confidence/error/upper bounds as None.
            error = jnp.asarray(0, dtype=frenet.dtype)
        elif error is None:
            error = jnp.asarray(jnp.nan, dtype=frenet.dtype)
        return chance_probabilities(
            frenet, scenarios.valid, scenarios.weights, distances, error,
            self.config.boundary_limit, self.config.safe_set_threshold,
        )

    def unavailable(self, reason, **metrics):
        if self.config.probabilistic_safety.fallback == "stop":
            raise SafetyStop("unavailable", detail=reason, **metrics)
        return {"probabilistic_safety_status": "unavailable", "chance_fallback": True,
                "chance_fallback_policy": "sit_conservative",
                "chance_unavailable_reason": reason, **metrics}

    def update(
        self,
        state,
        frenet_state,
        value_snapshot,
        safe_set,
        capture_debug=False,
        risk_snapshot=None,
        risk_cutoff=None,
        capture_risk=False,
    ):
        if value_snapshot is None:
            raise ValueError("learned controller requires a completed value snapshot")
        value_snapshot = inference_snapshot(value_snapshot)
        risk_snapshot = inference_snapshot(risk_snapshot)
        geometry_metrics = {}
        density_active = self.config.safe_set_query == "nf_density" and self.density_snapshot is not None
        if self.config.safe_set_query == "nf_density":
            if not density_active:
                raise ValueError(
                    "nf_density is selected but no fitted density snapshot is available; "
                    "the selected safe-set method will not be replaced."
                )
            if density_active and self._density_metadata_identity is not self.density_snapshot:
                self._density_metadata = dict(
                    safe_set_density_generation=int(self.density_snapshot.generation),
                    safe_set_density_cutoff=float(self.density_snapshot.cutoff),
                    safe_set_density_global_cutoff=float(self.density_snapshot.cutoff),
                    safe_set_density_calibration=self.config.safe_set_density.calibration,
                    safe_set_density_nll_min=float(self.density_snapshot.cutoff),
                    **{f"safe_set_density_{key}": value for key, value in
                       density_method_metadata(self.config.safe_set_density).items()},
                    safe_set_density_relaxation_cutoff=float(relaxation_cutoff(self.density_snapshot)),
                    safe_set_density_penalty_transform=("log1p" if self.density_snapshot.penalty_kind == "log1p" else "linear"),
                    safe_set_density_penalty_scale=float(self.density_snapshot.penalty_scale))
                if self.config.safe_set_density.method == "sigma":
                    # The sigma zero reference is a quantile, not the NLL minimum.
                    self._density_metadata.pop("safe_set_density_nll_min")
                if self.density_snapshot.fixed_penalty:
                    self._density_metadata["safe_set_density_upper_cutoff"] = float(
                        self.density_snapshot.cutoff + self.density_snapshot.penalty_scale)
                self._density_metadata_identity = self.density_snapshot
            geometry_metrics.update(safe_set_query_active="nf_density", **self._density_metadata)
        # The runner supplies host measurements. Preserve them before casting.
        speed = max(float(state[3]), 1.0)
        if isinstance(self.hull_evaluator, ImplicitHullEvaluator):
            self.hull_evaluator.reset_statistics()
        risk_active = self.config.risk_constraint.enabled and risk_cutoff is not None
        risk_distribution = self.config.risk_constraint.evaluation == "distribution"
        chance_active = self.config.probabilistic_safety.enabled
        provider = self.scenario_provider  # Freeze this belief for the entire update.
        empirical_chance = getattr(provider, "chance_estimation", None) == "planning_samples"
        independent_verification = (
            hasattr(provider, "verification_rollout")
            and getattr(provider, "uses_independent_verification", True)
        )
        chance_metrics = {}
        risk_unavailable_reason = None
        if risk_active:
            if risk_snapshot is None or risk_cutoff.get("status") != "ready":
                raise ValueError("active risk constraint requires a ready model/cutoff pair")
            if risk_cutoff.get("generation") != risk_snapshot.generation:
                raise ValueError("risk cutoff generation differs from risk model")
            cutoff = float(risk_cutoff["threshold"])
            if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
                raise ValueError("risk cutoff must be a finite probability")
        if provider is None:
            if chance_active:
                chance_metrics = self.unavailable("dynamics_distribution_unavailable")
                chance_active = False
            if risk_active and risk_distribution:
                risk_active = False
                risk_unavailable_reason = "dynamics_distribution_unavailable"
        diagnostic_risk = risk_snapshot is not None and (capture_risk or capture_debug)
        needs_endpoint = (
            value_snapshot.input_mode == "state_action" and not self.config.mean_value
            or risk_active and risk_snapshot.input_mode == "state_action"
        )
        # Inspection must not extend the policy's persistent action horizon.
        diagnostic_endpoint = (
            diagnostic_risk and risk_snapshot.input_mode == "state_action" and not needs_endpoint
        )
        dropping_endpoint = self.endpoint_action and not needs_endpoint
        retained_previous = self.actions[1:] if dropping_endpoint else None
        if needs_endpoint and not self.endpoint_action:
            self.actions = jnp.pad(self.actions, ((0, 1), (0, 0)))
        elif self.endpoint_action and not needs_endpoint:
            self.actions = self.actions[:self.config.horizon]
        self.endpoint_action = needs_endpoint
        state = jnp.asarray(state, dtype=self.model.params.dtype)
        previous = retained_previous if dropping_endpoint else jnp.concatenate([
            self.actions[1:], jnp.zeros((1, 2), dtype=self.actions.dtype)
        ])
        if self.randomness is None:
            self.key, sample_key, value_key = jax.random.split(self.key, 3)
        else:
            sample_key, value_key = self.randomness.next_key(), self.randomness.next_key()
        if hasattr(self.model, "resample"):
            self.model = self.model.resample(jax.random.fold_in(sample_key, 73))
            if provider is not None:
                provider = provider.with_model(self.model)
                self.scenario_provider = provider
        distribution_needed = chance_active or risk_active and risk_distribution
        shared_members = None
        if distribution_needed and getattr(provider, "can_reuse", lambda model: False)(self.model):
            cartesian, shared_members, da, sampled_actions = sample_predictions(
                self.model, state, previous, sample_key, self.config
            )
        else:
            cartesian, da, sampled_actions = sample_rollouts(
                self.model, state, previous, sample_key, self.config
            )
        speculative = window = None
        if not density_active and self.config.safe_set_selection == "predicted_window" and hasattr(safe_set, "speculate"):
            current_s = float(frenet_state[0])
            identity = (safe_set.signature,
                        id(getattr(self.model, "posterior_mean", None)),
                        id(getattr(self.model, "posterior_factor", None)),
                        id(self.model.bank) if isinstance(self.model, STEnsemble)
                        else id(self.model.functions.model) if hasattr(self.model, "functions")
                        else id(self.model.params))
            estimate = (current_s + speed * self.config.horizon * self.model.timestep,) * 2
            if self._window_prediction is not None:
                old_identity, old_s, old_bounds = self._window_prediction
                shift = current_s - old_s
                if identity == old_identity and abs(shift) <= max(5., speed * self.model.timestep * 4):
                    estimate = (old_bounds[0] + shift, old_bounds[1] + shift)
            lo, hi = safe_set.window(estimate, speed)
            margin = self.config.safe_set_window_margin
            window = lo - margin, hi + margin
            try:
                # No GPU values are read: CPU selection/Qhull overlaps rollouts.
                speculative = safe_set.speculate(window, self.model.params.dtype)
            except ValueError:
                geometry_metrics["safe_set_speculation"] = "construction_failed"
        indices = jnp.asarray(
            self.track.get_s_search_range(frenet_state[0], self.config.search_points)
        )
        states = to_frenet(self.track, cartesian, indices, frenet_state[0])
        scenario_key = jax.random.fold_in(sample_key, 17) if distribution_needed else None
        sample_scenarios = sample_frenet = sample_chance = None
        if distribution_needed:
            sample_scenarios, sample_frenet = self.scenarios(
                provider, state, sampled_actions, scenario_key, indices, frenet_state[0], shared_members, states
            )
            if not bool(jnp.all(distribution_availability(
                sample_frenet, sample_scenarios.valid, sample_scenarios.weights
            ))):
                if chance_active:
                    chance_metrics = self.unavailable("invalid_sample_scenarios", **provider.metadata)
                    chance_active = False
                if risk_active and risk_distribution:
                    risk_active = False
                    risk_unavailable_reason = "invalid_sample_scenarios"
                distribution_needed = False
        if density_active and self.config.safe_set_density.calibration == "local":
            lo, hi, finite = jax.device_get(terminal_summary(states))
            if not finite:
                raise FloatingPointError("nonfinite learned-controller samples")
            cache = self.density_calibration
            if cache is not None and cache.snapshot is self.density_snapshot:
                local_hull, calibration_metrics = cache.select(
                    safe_set, (float(lo), float(hi)), speed, self.config.safe_set_density
                )
            else:
                local_hull, calibration_metrics = None, dict(
                    safe_set_density_calibration="local",
                    safe_set_density_calibration_status="missing_or_stale_calibration",
                )
            geometry_metrics.update(calibration_metrics)
            if local_hull is None:
                reason = calibration_metrics["safe_set_density_calibration_status"]
                raise ValueError(
                    f"nf_density local calibration is unavailable: {reason}; "
                    "the selected safe-set method will not be replaced."
                )
        if density_active:
            # Freeze one calibration pair for samples, proposals and all scenarios.
            hull = local_hull if self.config.safe_set_density.calibration == "local" else self.density_snapshot
            selected, reference = safe_set.arrays() if capture_debug else (np.empty((0, 8)), None)
        elif hasattr(safe_set, "select_bounds"):
            lo, hi, finite = jax.device_get(terminal_summary(states))
            if not finite:
                raise FloatingPointError("nonfinite learned-controller samples")
            bounds = float(lo), float(hi)
            needed = safe_set.window(bounds, speed)
            if (speculative is not None and speculative[-1] == safe_set.signature
                    and window[0] <= needed[0] and needed[1] <= window[1]):
                selected, reference, hull, rng_state, _ = speculative
                safe_set.rng.bit_generator.state = rng_state
                geometry_metrics["safe_set_speculation"] = "accepted"
            else:
                selected, reference = safe_set.select_bounds(bounds, speed)
                hull = build_safe_geometry(selected[:, :5], self.config, dtype=self.model.params.dtype)
                if window is not None:
                    geometry_metrics.setdefault("safe_set_speculation", "window_miss")
            if window is not None:
                self._window_prediction = identity, current_s, bounds
                geometry_metrics.update(safe_set_window_min=window[0], safe_set_window_max=window[1],
                                        safe_set_actual_min=needed[0], safe_set_actual_max=needed[1])
        else:
            # Preserve duck-typed/custom selectors; only built-in selection uses bounds.
            terminals = np.asarray(jax.device_get(states[:, -1, :5]))
            if not np.isfinite(terminals).all():
                raise FloatingPointError("nonfinite learned-controller samples")
            selected, reference = safe_set.select(terminals, speed)
            hull = build_safe_geometry(selected[:, :5], self.config, dtype=self.model.params.dtype)
        if value_snapshot.input_mode == "state_action":
            value_states, value_actions = value_query_inputs(
                states,
                jnp.asarray(frenet_state, dtype=states.dtype),
                sampled_actions,
                self.model.action_scale,
                self.config.mean_value,
            )
            values = predict_value(
                value_snapshot, value_states, value_key, self.value_samples, value_actions
            )
        else:
            values = predict_value(
                value_snapshot, states.reshape(-1, 7), value_key, self.value_samples
            )
        density_normalization = None
        if (isinstance(hull, DensitySnapshot) and hull.fixed_penalty
                and self.config.safe_set_density.penalty_normalization == "sample_max"):
            valid = sample_validity(previous, da, states, values, jnp.zeros(self.config.samples))
            hull, sample_distances, density_normalization = sample_normalized_density(hull, states[:, -1, :5], valid)
        else:
            sample_distances = self.evaluate_hull(states[:, -1, :5], hull)
        sample_risk = None
        risk_started = time.perf_counter() if risk_active else None
        if chance_active:
            sample_chance = self.chance(
                sample_scenarios, sample_frenet, hull,
                empirical=hasattr(provider, "verification_rollout"),
            )
            if not bool(jnp.all(sample_chance["available"])):
                chance_metrics = self.unavailable("probability_estimate_unavailable", **provider.metadata)
                chance_active = False
                sample_chance = None
        if risk_active:
            sample_risk = (
                distribution_risk_probabilities(
                    risk_snapshot, sample_scenarios, sample_frenet, sampled_actions, self.model.action_scale
                ) if risk_distribution else rollout_risk_probabilities(
                    risk_snapshot, states, sampled_actions, self.model.action_scale
                )
            )
            if not bool(jnp.all(jnp.isfinite(sample_risk))):
                raise FloatingPointError("nonfinite sampled risk probabilities")
        active_lambdas = self.active_lambdas(chance_active, risk_active,
                                           force_hull=not isinstance(hull, DensitySnapshot))
        fixed_density_cost = isinstance(hull, DensitySnapshot) and hull.fixed_penalty
        proposal_options = {"fixed_density_cost": True} if fixed_density_cost else {}
        if chance_active:
            candidates = propose_with_constraints(
                previous, da, states, values, sample_distances,
                {name: sample_chance[name] for name in ("boundary", "terminal_hull")},
                sample_risk, active_lambdas, self.config, self.config.risk_constraint.mode,
                **proposal_options,
            )
        elif risk_active:
            candidates = propose_with_risk(
                previous, da, states, values, sample_distances, sample_risk,
                active_lambdas, self.config, self.config.risk_constraint.mode,
                **proposal_options,
            )
        else:
            candidates = propose(
                previous, da, states, values, sample_distances, active_lambdas, self.config, **proposal_options
            )
        trajectories, member_trajectories = candidate_predictions(self.model, state, candidates, self.config)
        candidate_frenet = to_frenet(self.track, trajectories, indices, frenet_state[0])
        candidate_distances = self.evaluate_safe_set_check(candidate_frenet[:, -1, :5], hull)
        candidate_scenarios = scenario_frenet = candidate_chance = verified_scenarios = None
        if chance_active or risk_active and risk_distribution:
            candidate_scenarios, scenario_frenet = self.scenarios(
                provider, state, candidates, scenario_key, indices, frenet_state[0], member_trajectories, candidate_frenet
            )
            distribution_valid = bool(jnp.all(distribution_availability(
                scenario_frenet, candidate_scenarios.valid, candidate_scenarios.weights
            )))
            if not distribution_valid:
                if chance_active:
                    chance_metrics = self.unavailable("invalid_candidate_scenarios", **provider.metadata)
                    chance_active = False
                if risk_active and risk_distribution:
                    # Scoring/gating is unavailable; do not pretend it is zero.
                    risk_active = False
                    risk_unavailable_reason = "invalid_candidate_scenarios"
            if chance_active:
                verified_scenarios, verification_frenet = candidate_scenarios, scenario_frenet
                if independent_verification:
                    # Candidate controls and hull are fixed before this reserved IID batch.
                    verified_scenarios, verification_frenet = self.scenarios(
                        provider, state, candidates, None, indices, frenet_state[0],
                        anchor=candidate_frenet,
                        supplied=provider.verification_rollout(
                            state, candidates[:, :self.config.horizon],
                            planning_key=scenario_key,
                            validation_key=jax.random.fold_in(sample_key, 91),
                        ),
                    )
                candidate_chance = self.chance(
                    verified_scenarios, verification_frenet, hull, empirical=empirical_chance
                )
                if not bool(jnp.all(candidate_chance["available"])):
                    chance_metrics = self.unavailable("candidate_probability_estimate_unavailable", **provider.metadata)
                    chance_active = False
                    candidate_chance = None
        terminal_values = None
        if value_snapshot.input_mode == "state_action" and not self.config.mean_value:
            query_states, query_actions = endpoint_query_inputs(
                candidate_frenet, candidates, self.model.action_scale
            )
            terminal_values = predict_value(
                value_snapshot,
                query_states,
                jax.random.fold_in(value_key, 1),
                self.value_samples,
                query_actions,
            )
        candidate_risk = None
        if risk_active:
            candidate_risk = (
                distribution_risk_probabilities(
                    risk_snapshot, candidate_scenarios, scenario_frenet, candidates, self.model.action_scale
                ) if risk_distribution else rollout_risk_probabilities(
                    risk_snapshot, candidate_frenet, candidates, self.model.action_scale
                )
            )
            if not bool(jnp.all(jnp.isfinite(candidate_risk))):
                raise FloatingPointError("nonfinite candidate risk probabilities")
        packet = finish_candidates(
            candidate_frenet, candidate_distances, candidate_chance if chance_active else None,
            candidate_risk if risk_active else None, cutoff if risk_active else None,
            terminal_values, candidates, trajectories,
            member_trajectories if hasattr(self.model, "weights") else None,
            self.model.weights if hasattr(self.model, "weights") else None,
            active_lambdas, self.config,
        )
        packet["valid_sample_count"] = jnp.sum(sample_validity(previous, da, states, values, sample_distances))
        if density_normalization is not None:
            packet["density_normalization"] = density_normalization
        # Keep the chosen horizon device-resident. Download all required fields once.
        if isinstance(self.hull_evaluator,ImplicitHullEvaluator):
            packet["implicit_hull_diagnostics"] = self.hull_evaluator.diagnostics()
        retained_actions = packet["actions"]
        packet, host_hull_mask, error_bound = jax.device_get((
            packet, hull.mask,
            verified_scenarios.probability_error_bound
            if chance_active and not empirical_chance else None,
        ))
        index, ss_violation, boundary_violation = packet["index"], packet["hull"], packet["boundary"]
        actions, trajectory, lambdas = packet["actions"], packet["trajectory"], packet["lambdas"]
        terminal_value = packet["terminal_value"]
        valid_sample_count, finite_candidate_count = int(packet["valid_sample_count"]), int(packet["finite_candidate_count"])
        if not valid_sample_count or not finite_candidate_count:
            raise FloatingPointError(
                f"learned controller has {valid_sample_count}/{self.config.samples} valid samples and "
                f"{finite_candidate_count}/{len(candidates)} finite candidates; "
                f"speed={float(state[3]):.4g} m/s"
            )
        if density_normalization is not None:
            geometry_metrics.update({name: float(value) for name, value in packet["density_normalization"].items()})
            geometry_metrics["safe_set_density_penalty_normalization"] = "sample_max"
        geometry_metrics.update({key: np.asarray(value).item()
            for key, value in packet.get("implicit_hull_diagnostics", {}).items()})
        model_trajectories, probabilities = packet["members"], packet["probabilities"]
        chosen_risk, feasible_count, fallback = packet["chosen_risk"], packet["feasible_count"], packet["fallback"]
        if chance_active:
            count, used_fallback = feasible_count, fallback
            if used_fallback and self.config.probabilistic_safety.fallback == "stop":
                raise SafetyStop(
                    "no_feasible_candidate", feasible_candidates=int(count),
                    boundary_threshold=self.config.probabilistic_safety.boundary_threshold,
                    terminal_hull_threshold=self.config.probabilistic_safety.terminal_hull_threshold,
                    minimum_boundary_probability=float(packet["minimum_boundary"]),
                    minimum_terminal_hull_probability=float(packet["minimum_hull"]),
                    **provider.metadata,
                )
            chosen = packet["chosen_chance"]
            chance_metrics = {
                "probabilistic_safety_status": "no_feasible_candidate" if used_fallback else "active",
                "chance_fallback": bool(used_fallback),
                "chance_fallback_policy": "sit_conservative" if used_fallback else None,
                "boundary_probability": float(chosen["boundary"]),
                "terminal_hull_probability": float(chosen["terminal_hull"]),
                # Empirical gates use raw fractions, not statistical upper bounds.
                "boundary_probability_upper": None if empirical_chance else float(chosen["boundary_upper"]),
                "terminal_hull_probability_upper": None if empirical_chance else float(chosen["terminal_hull_upper"]),
                "boundary_probability_threshold": self.config.probabilistic_safety.boundary_threshold,
                "terminal_hull_probability_threshold": self.config.probabilistic_safety.terminal_hull_threshold,
                "chance_feasible_candidates": int(packet["chance_count"]),
                "joint_feasible_candidates": int(count),
                "chance_probability_error_bound": None if empirical_chance else float(error_bound),
                **provider.metadata,
            }
            if not used_fallback and (
                not bool(chosen["available"])
                or chosen["boundary_upper"] > self.config.probabilistic_safety.boundary_threshold
                or chosen["terminal_hull_upper"] > self.config.probabilistic_safety.terminal_hull_threshold
            ):
                raise AssertionError("selected optimized sequence exceeds probability thresholds")
        if not np.isfinite(actions).all() or not np.isfinite(trajectory).all():
            raise FloatingPointError("nonfinite learned-controller candidate")
        if terminal_value is not None and not np.isfinite(terminal_value):
            raise FloatingPointError("nonfinite learned-controller terminal value")
        risk_metrics = {}
        if risk_active:
            # This is the final, already rerolled weighted candidate; do not
            # infer its feasibility from the originally sampled paths.
            if not bool(fallback) and float(chosen_risk) > cutoff:
                raise AssertionError("selected risk candidate exceeds its cutoff")
            risk_failed = float(chosen_risk) > cutoff
            risk_metrics = {
                "risk_constraint_status": "no_feasible_candidate" if risk_failed else "active",
                "risk_generation": int(risk_snapshot.generation),
                "risk_cutoff": cutoff,
                "risk_mode": self.config.risk_constraint.mode,
                "risk_evaluation": self.config.risk_constraint.evaluation,
                "risk_score": float(chosen_risk),
                "risk_violation": max(float(chosen_risk) - cutoff, 0.0),
                "risk_feasible_candidates": int(packet["risk_count"]),
                "risk_fallback": risk_failed,
                "risk_constraint_seconds": time.perf_counter() - risk_started,
                **(provider.metadata if risk_distribution else {}),
            }
        elif self.config.risk_constraint.enabled:
            risk_metrics = {"risk_constraint_status": "unavailable",
                            "risk_unavailable_reason": risk_unavailable_reason or "model_or_cutoff_unavailable"}
        self.actions = retained_actions
        diagnostic_actions = sampled_actions
        if diagnostic_endpoint:
            # Reuse the folded endpoint draw without advancing the policy RNG.
            _, extended_actions = sample_actions(
                jnp.pad(previous, ((0, 1), (0, 0))), sample_key, self.config
            )
            diagnostic_actions = jnp.concatenate(
                (sampled_actions, extended_actions[:, -1:]), axis=1
            )
        risk = None
        if diagnostic_risk:
            risk = (
                jax.device_get(
                    {
                        "probabilities": sample_risk,
                        "frenet": states[:, :, :2],
                        "generation": risk_snapshot.generation,
                        "input_mode": risk_snapshot.input_mode,
                    }
                )
                if risk_active
                else sampled_risk_diagnostic(
                    risk_snapshot, states, diagnostic_actions, self.model.action_scale
                )
            )
        debug = None
        if capture_debug:
            # Returning extra arrays from propose can change GPU fusion/rounding.
            # Keep the policy kernel identical and recompute only deterministic
            # plot terms from this update; never resample or overwrite candidates.
            terms = debug_reward_terms(
                previous, da, states, values, sample_distances, active_lambdas, self.config, **proposal_options
            )
            debug_reward = terms["reward"][index]
            if risk_active:
                debug_reward = debug_reward - float(lambdas[-1]) * normalized_risk_cost(
                    sample_risk, self.config.risk_constraint.mode
                )
            if chance_active:
                terminal = jnp.zeros(self.config.horizon, dtype=states.dtype).at[-1].set(1)
                debug_reward -= (float(lambdas[2]) * sample_chance["boundary"]
                                 + float(lambdas[3]) * sample_chance["terminal_hull"])[:, None] * terminal
            debug_weights = weights(returns(debug_reward), self.config.temperature, self.config.damping,
                                    valid=terms["valid"][:, None])
            scores, ss, boundaries, finite = candidate_scores(
                candidate_frenet, candidate_distances, self.config.boundary_limit,
                self.config.safe_set_threshold, terminal_values,
            )
            finite &= (jnp.all(jnp.isfinite(candidates), axis=(1, 2))
                       & jnp.all(jnp.isfinite(trajectories), axis=(1, 2)))
            debug_endpoint_action = actions[-1]
            if diagnostic_endpoint:
                terminal_weights = weights(
                    debug_reward[:, -1], self.config.temperature, self.config.damping, axis=0,
                    valid=terms["valid"],
                )
                debug_endpoint_action = jnp.clip(jnp.sum(
                    terminal_weights[:, None] * diagnostic_actions[:, -1], axis=0
                ), -1, 1)
            debug = jax.device_get(
                {
                    "states": states,
                    "sample_cartesian": cartesian,
                    "valid_samples": terms["valid"],
                    "safe_set": selected,
                    "candidate_states": candidate_frenet,
                    "candidate_cartesian": trajectories,
                    "candidate_actions": candidates,
                    "candidate_scores": jnp.where(finite, scores, jnp.inf),
                    "candidate_safe_set": ss,
                    "candidate_boundary_sum": boundaries,
                    "candidate_valid": finite,
                    "sample_actions": sampled_actions[:, : self.config.horizon],
                    "previous_actions": previous[: self.config.horizon],
                    "proposal_previous": previous,
                    "proposal_noise": da,
                    "proposal_values": values,
                    "proposal_lambdas": active_lambdas,
                    "planning_state": state,
                    "planning_frenet": jnp.asarray(frenet_state, state.dtype),
                    "planning_s": jnp.asarray(frenet_state[0], state.dtype),
                    "planning_search_indices": indices,
                    "chosen_actions": self.actions[: self.config.horizon],
                    **(
                        {
                            "sample_endpoint_actions": diagnostic_actions[:, -1],
                            "chosen_endpoint_action": debug_endpoint_action,
                        }
                        if self.endpoint_action or diagnostic_endpoint
                        else {}
                    ),
                    "chosen_states": candidate_frenet[index],
                    "candidate_boundary": jnp.where(
                        jnp.abs(candidate_frenet[:, :, 1]) > self.config.boundary_limit,
                        jnp.abs(candidate_frenet[:, :, 1]),
                        0,
                    ),
                    "reward": debug_reward,
                    "sample_weights": debug_weights,
                    "value": terms["value"],
                    "hull": terms["hull"],
                    "boundary": terms["boundary"],
                    **(
                        {"risk_probabilities": sample_risk, "candidate_risk": candidate_risk}
                        if risk_active else {}
                    ),
                    **({"sample_chance": sample_chance, "candidate_chance": candidate_chance}
                       if chance_active else {}),
                    **({
                        "sample_nll": density_scores(hull, states[:, -1, :5]),
                        "candidate_nll": density_scores(hull, candidate_frenet[:, -1, :5]),
                        "safe_set_nll": (density_scores(hull, jnp.asarray(selected[:, :5], state.dtype))
                                         if len(selected) else jnp.zeros(0, state.dtype)),
                        "density_cutoff": hull.cutoff,
                        "density_relaxation_cutoff": relaxation_cutoff(hull),
                        "density_upper": jnp.asarray(geometry_metrics.get("safe_set_density_upper_cutoff",
                                                                          hull.cutoff + hull.penalty_scale), state.dtype),
                        "density_penalty_scale": hull.penalty_scale,
                        "density_penalty_normalizer": jnp.asarray(hull.penalty_normalizer, state.dtype),
                        "density_offset": hull.offset,
                        "density_scale": hull.scale,
                        **({"density_window": jnp.asarray([hull.window_min, hull.window_max])}
                           if hasattr(hull, "window_min") else {}),
                    } if isinstance(hull, DensitySnapshot) else {}),
                }
            )
            debug["risk_mode"] = self.config.risk_constraint.mode
            debug["safe_set_selection_speed"] = speed
            debug["sample_hull_distances"] = np.asarray(jax.device_get(sample_distances))
        return PlannerResult(
            actions[0],
            trajectory,
            reference,
            float(ss_violation),
            float(boundary_violation),
            tuple(map(float, lambdas)),
            {
                "candidate": int(index),
                "invalid_samples": self.config.samples - valid_sample_count,
                "nonfinite_candidates": len(candidates) - finite_candidate_count,
                "safe_set_query": self.config.safe_set_query,
                "hull_facets": int(host_hull_mask.sum()) if self.config.safe_set_query=="facets" else None,
                "hull_padded_facets": len(hull.mask) if self.config.safe_set_query=="facets" else None,
                **({"hull_points":int(host_hull_mask.sum()),"hull_padded_points":len(hull.mask)}
                   if self.config.safe_set_query=="implicit" else {}),
                "value_generation": int(value_snapshot.generation),
                **({"terminal_value": float(terminal_value)} if terminal_value is not None else {}),
                **risk_metrics,
                **chance_metrics,
                **geometry_metrics,
                "lambda_terms": ["hull", "boundary"]
                + (["chance_boundary", "chance_hull"] if sample_chance is not None else [])
                + (["risk"] if sample_risk is not None else []),
            },
            debug,
            risk,
            model_trajectories,
            probabilities,
        )
