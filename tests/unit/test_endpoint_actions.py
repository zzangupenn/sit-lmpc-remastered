from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config import ExperimentConfig
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.planners import lmppi
from sit_lmpc.planners.common import returns, unit_range, weights
from sit_lmpc.training import value


def fixture():
    config = ExperimentConfig()
    planner = replace(config.planner, horizon=3, samples=8, lambda_count=2, hull_precompile=False)
    _, model = build_vehicle_dynamics(config)
    state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=model.params.dtype)
    return planner, model, state


@pytest.mark.parametrize("scan", [False, True])
def test_endpoint_is_sampled_bounded_and_never_integrated(scan):
    config, model, state = fixture()
    config = replace(config, scan_rollout=scan)
    previous = jnp.zeros((config.horizon, 2), dtype=model.params.dtype)
    key = jax.random.key(81)
    baseline_states, baseline_da, baseline_actions = lmppi.sample_rollouts(
        model, state, previous, key, config
    )
    states, da, actions = lmppi.sample_rollouts(
        model, state, jnp.pad(previous, ((0, 1), (0, 0))), key, config
    )
    assert states.shape == (8, 3, 7) and actions.shape == (8, 4, 2)
    np.testing.assert_array_equal(states, baseline_states)
    np.testing.assert_array_equal(actions[:, :3], baseline_actions)
    np.testing.assert_array_equal(da[:, :3], baseline_da)
    assert np.max(np.abs(actions)) <= 1 and np.std(actions[:, -1]) > 0
    candidates = actions[:4]
    trajectory = lmppi.candidate_rollouts(model, state, candidates, config)
    changed = candidates.at[:, -1].set(1)
    np.testing.assert_array_equal(
        trajectory, lmppi.candidate_rollouts(model, state, changed, config)
    )
    endpoint, physical = lmppi.endpoint_query_inputs(trajectory, candidates, model.action_scale)
    np.testing.assert_array_equal(endpoint, trajectory[:, -1])
    np.testing.assert_allclose(physical, candidates[:, -1] * model.action_scale)
    with pytest.raises(ValueError, match="H\\+1"):
        lmppi.endpoint_query_inputs(trajectory, candidates[:, :3], model.action_scale)


def test_terminal_value_updates_endpoint_with_same_sequence_weights():
    config, model, _ = fixture()
    config = replace(config, mean_value=False, action_rate_penalty=0)
    previous = jnp.zeros((4, 2), dtype=model.params.dtype)
    # Identical physical horizons and states; only u_H differs.
    da = (
        jnp.zeros((8, 4, 2), dtype=model.params.dtype)
        .at[:, -1, 0]
        .set(jnp.linspace(-0.8, 0.8, 8, dtype=model.params.dtype))
    )
    states = jnp.zeros((8, 3, 7), dtype=model.params.dtype)
    values = da[:, -1, 0]
    distances = jnp.zeros(8, dtype=model.params.dtype)
    lambdas = jnp.zeros((1, 2), dtype=model.params.dtype)
    proposed = lmppi.propose(previous, da, states, values, distances, lambdas, config)
    reward = jnp.zeros((1, 8, 3), dtype=model.params.dtype).at[0, :, -1].set(unit_range(values))
    w = weights(returns(reward), config.temperature, config.damping)
    expected = jnp.sum(w[0, :, -1, None] * da[:, -1], axis=0)
    np.testing.assert_allclose(proposed[0, -1], expected, rtol=1e-6)
    assert proposed[0, -1, 0] > 0.1
    np.testing.assert_array_equal(proposed[:, :3], 0)
    reversed_value = lmppi.propose(previous, da, states, -values, distances, lambdas, config)
    assert reversed_value[0, -1, 0] < -0.1


@pytest.mark.parametrize("mean", [False, True])
def test_state_only_proposal_matches_original_formula(mean):
    config, model, state = fixture()
    config = replace(config, mean_value=mean)
    previous = jnp.zeros((3, 2), dtype=model.params.dtype)
    states, da, _ = lmppi.sample_rollouts(model, state, previous, jax.random.key(3), config)
    values = jnp.arange(24, dtype=model.params.dtype)
    distances = jnp.linspace(0, 1, 8, dtype=model.params.dtype)
    lambdas = jnp.asarray([[0, 0], [1, 2]], dtype=model.params.dtype)

    @jax.jit
    def original(previous, da, states, values, distances, lambdas):
        terminal = jnp.zeros(config.horizon).at[-1].set(1)
        boundary = jnp.abs(states[:, :, 1])
        r_boundary = unit_range(jnp.where(boundary > config.boundary_limit, boundary, 0))
        r_value = values.reshape(config.samples, config.horizon)
        if config.mean_value:
            r_value = r_value.mean(axis=1, keepdims=True)
        r_value = unit_range(r_value) * terminal
        distance = jnp.where(
            distances > config.safe_set_relaxation, distances - config.safe_set_relaxation, 0
        )
        r_hull = unit_range(jnp.minimum(distance, config.safe_set_reward_limit))[:, None] * terminal
        changes = jnp.linalg.norm((da + previous)[:, 1:] - (da + previous)[:, :-1], axis=-1)
        r_action = jnp.pad(changes, ((0, 0), (1, 0))) * config.action_rate_penalty
        reward = (
            r_value[None]
            - lambdas[:, 0, None, None] * r_hull[None]
            - lambdas[:, 1, None, None] * r_boundary[None]
            - r_action[None]
        )
        w = weights(returns(reward), config.temperature, config.damping)
        return jnp.clip(previous + jnp.sum(w[:, :, :, None] * da[None], axis=1), -1, 1)

    args = previous, da, states, values, distances, lambdas
    np.testing.assert_array_equal(lmppi.propose(*args, config), original(*args))


def test_rechecked_value_does_not_override_sit_first_lambda_preference():
    states = jnp.zeros((3, 3, 7))
    distances = jnp.asarray([0, 0, 1.0])
    values = jnp.asarray([1, 2, 1000])
    assert int(lmppi.select_candidate(states, distances, 3.0, 0.1)[0]) == 0
    assert int(lmppi.select_candidate(states, distances, 3.0, 0.1, values)[0]) == 0


@pytest.mark.parametrize(
    "value_mode,risk_mode",
    [
        ("state", "state"),
        ("state", "state_action"),
        ("state_action", "state"),
        ("state_action", "state_action"),
    ],
)
@pytest.mark.parametrize("mean", [False, True])
def test_planner_preserves_endpoint_candidate_pair_and_shifts_warm_start(
    monkeypatch, value_mode, risk_mode, mean
):
    config, model, state = fixture()
    config = replace(config, mean_value=mean)
    track = SimpleNamespace(get_s_search_range=lambda *args: np.arange(2))
    planner = lmppi.LMPPI(
        config, model, track, value_input_mode=value_mode, risk_input_mode=risk_mode
    )
    monkeypatch.setattr(lmppi, "to_frenet", lambda track, states, *args: states)
    monkeypatch.setattr(
        lmppi, "build_safe_geometry", lambda *args, **kwargs: SimpleNamespace(mask=np.ones(6))
    )
    planner.evaluate_hull = lambda states, hull: jnp.zeros(len(states))
    safe = SimpleNamespace(select=lambda *args: (np.zeros((6, 8)), np.zeros((2, 7))))
    queries, previous = [], []
    risk_queries = []
    original_sample = lmppi.sample_rollouts

    def sample(model, state, warm_start, key, config):
        previous.append(np.asarray(warm_start).copy())
        return original_sample(model, state, warm_start, key, config)

    def predict(snapshot, states, key, samples, actions=None):
        queries.append((np.asarray(states), None if actions is None else np.asarray(actions)))
        return states[:, 0] if actions is None else actions[:, 1]

    monkeypatch.setattr(lmppi, "sample_rollouts", sample)
    monkeypatch.setattr(lmppi, "predict_value", predict)

    def predict_risk(snapshot, states, actions=None):
        risk_queries.append((np.asarray(states), None if actions is None else np.asarray(actions)))
        return jnp.linspace(0, 1, len(states))

    monkeypatch.setattr(lmppi, "predict_risk", predict_risk)
    snapshot = SimpleNamespace(input_mode=value_mode, generation=1)
    risk_snapshot = SimpleNamespace(input_mode=risk_mode, generation=3)
    result = planner.update(
        state,
        np.asarray(state),
        snapshot,
        safe,
        capture_debug=True,
        risk_snapshot=risk_snapshot,
    )
    extra = value_mode == "state_action" and not mean
    assert planner.actions.shape == (3 + extra, 2)
    assert result.trajectory.shape == (3, 7)
    assert result.debug["sample_actions"].shape == (8, 3, 2)
    assert result.debug["chosen_actions"].shape == (3, 2)
    assert result.risk["probabilities"].shape == (8, 3)
    assert result.risk["frenet"].shape == (8, 3, 2)
    assert result.risk["generation"] == 3
    np.testing.assert_array_equal(risk_queries[0][0].reshape(8, 3, 7), result.debug["states"])
    np.testing.assert_array_equal(result.risk["probabilities"].ravel(), jnp.linspace(0, 1, 24))
    if risk_mode == "state_action":
        samples = np.concatenate(
            (
                result.debug["sample_actions"][:, 1:],
                result.debug["sample_endpoint_actions"][:, None],
            ),
            axis=1,
        )
        np.testing.assert_allclose(
            risk_queries[0][1].reshape(8, 3, 2), samples * model.action_scale
        )
    else:
        assert risk_queries[0][1] is None
    if value_mode == "state_action" and not mean:
        assert len(queries) == 2 and queries[0][0].shape == (8, 7)
        candidate = result.metrics["candidate"]
        np.testing.assert_array_equal(queries[1][0][candidate], result.trajectory[-1])
        np.testing.assert_allclose(
            queries[1][1][candidate], planner.actions[-1] * model.action_scale
        )
        assert result.metrics["terminal_value"] == pytest.approx(queries[1][1][candidate, 1])
    else:
        assert len(queries) == 1 and queries[0][0].shape == (24, 7)
    chosen = np.asarray(planner.actions).copy()
    planner.update(state, np.asarray(state), snapshot, safe)
    assert len(risk_queries) == 1
    np.testing.assert_array_equal(previous[1][:-1], chosen[1:])
    np.testing.assert_array_equal(previous[1][-1], 0)


def test_value_publication_warms_both_fixed_batch_sizes(monkeypatch):
    trainer = object.__new__(value.ValueTrainer)
    trainer.prediction_batch_size = (8, 4, 8)
    trainer.seed = 123
    trainer.config = SimpleNamespace(samples=2)
    snapshot = SimpleNamespace(
        input_mode="state_action", normalization=SimpleNamespace(low=jnp.zeros(8))
    )
    calls = []

    def predict(snapshot, states, key, samples, actions):
        calls.append((states.shape, actions.shape))
        return jnp.zeros(len(states))

    monkeypatch.setattr(value, "predict_value", predict)
    trainer.warmup(snapshot)
    assert calls == [((8, 7), (8, 2)), ((4, 7), (4, 2))]
