"""Stage 4 learned-risk proposal and final weighted-candidate gate."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import ExperimentConfig, RiskConstraintConfig, load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.planners import lmppi


def fixture():
    config = ExperimentConfig()
    constraint = RiskConstraintConfig(
        enabled=True, mode="terminal", lambda_count=2, lambda_max=8.0, max_candidates=4
    )
    planner_config = replace(
        config.planner, horizon=3, samples=8, lambda_count=2, hull_precompile=False,
        risk_constraint=constraint,
    )
    _, model = build_vehicle_dynamics(config)
    state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=model.params.dtype)
    return planner_config, model, state


def test_constraint_config_is_explicit_and_requires_risk_training():
    config = load_config(overrides={"risk": {"enabled": True}, "planner": {"risk_constraint": {"enabled": True, "mode": "all"}}})
    assert config.planner.risk_constraint.enabled
    assert config.planner.risk_constraint.mode == "all"
    for changes in (
        {"mode": "middle"},
        {"lambda_count": 0},
        {"max_candidates": 0},
        {"lambda_max": float("nan")},
    ):
        with pytest.raises(ValueError):
            load_config(overrides={"planner": {"risk_constraint": changes}})
    with pytest.raises(ValueError, match="requires risk.enabled"):
        load_config(
            overrides={
                "planner": {"risk_constraint": {"enabled": True}},
                "risk": {"enabled": False},
            }
        )


def test_risk_cost_normalizes_raw_probability_and_uses_each_all_mode_step():
    config, model, _ = fixture()
    config = replace(config, temperature=0.1, action_rate_penalty=0)
    previous = jnp.zeros((3, 2), dtype=model.params.dtype)
    da = jnp.zeros((8, 3, 2), dtype=model.params.dtype)
    da = da.at[:4, :, 0].set(-0.5).at[4:, :, 0].set(0.5)
    states = jnp.zeros((8, 3, 7), dtype=model.params.dtype)
    values = jnp.zeros(24, dtype=model.params.dtype)
    distances = jnp.zeros(8, dtype=model.params.dtype)
    lambdas = jnp.asarray([[0, 0, 0], [0, 0, 8]], dtype=model.params.dtype)
    risk = jnp.full((8, 3), 0.1, dtype=model.params.dtype)
    risk = risk.at[4:, 0].set(0.29).at[4:, 1].set(0.25)
    normalized_all = np.asarray(lmppi.normalized_risk_cost(risk, "all"))
    normalized_terminal = np.asarray(lmppi.normalized_risk_cost(risk, "terminal"))
    assert normalized_all.min() == pytest.approx(0)
    assert normalized_all.max() == pytest.approx(1)
    np.testing.assert_array_equal(normalized_all[:4], np.zeros((4, 3)))
    np.testing.assert_allclose(normalized_all[4:, 0], 1)
    np.testing.assert_allclose(normalized_all[4:, 1], (0.25 - 0.1) / (0.29 - 0.1))
    np.testing.assert_allclose(normalized_all[:, 2], 0)
    np.testing.assert_array_equal(normalized_terminal, np.zeros((8, 3)))
    probe = jnp.asarray([[0.1, 0.1, 0.1], [0.1, 0.4, 0.1], [0.1, 0.1, 0.7]])
    np.testing.assert_allclose(
        np.asarray(lmppi.normalized_risk_cost(probe, "all")),
        (np.asarray(probe) - 0.1) / 0.6,
    )
    np.testing.assert_allclose(
        np.asarray(lmppi.normalized_risk_cost(probe, "terminal"))[:, -1], [0, 0, 1]
    )
    terminal = lmppi.propose_with_risk(
        previous, da, states, values, distances, risk, lambdas, config, "terminal"
    )
    all_steps = lmppi.propose_with_risk(
        previous, da, states, values, distances, risk, lambdas, config, "all"
    )
    np.testing.assert_allclose(terminal[0], terminal[1], atol=1e-6)
    np.testing.assert_allclose(all_steps[0], terminal[0], atol=1e-6)
    assert float(all_steps[1, 0, 0]) < -0.4
    assert float(all_steps[1, 1, 0]) < -0.4
    assert float(all_steps[1, 2, 0]) == pytest.approx(0, abs=1e-6)


def test_final_candidate_risk_gate_and_explicit_baseline_fallback():
    trajectories = jnp.zeros((3, 3, 7))
    trajectories = trajectories.at[:, :, 1].set(jnp.asarray([0.0, 0.5, 1.0])[:, None])
    distances = jnp.zeros(3)
    risk = jnp.asarray([[0.9, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.1, 0.9]])
    terminal = lmppi.select_risk_candidate(
        trajectories, distances, risk, 0.3, "terminal", 3.0, 0.03
    )
    all_steps = lmppi.select_risk_candidate(
        trajectories, distances, risk, 0.3, "all", 3.0, 0.03
    )
    assert int(terminal[0]) == 0 and int(terminal[4]) == 2 and not bool(terminal[5])
    assert int(all_steps[0]) == 1 and int(all_steps[4]) == 1 and not bool(all_steps[5])
    fallback = lmppi.select_risk_candidate(
        trajectories, distances, jnp.ones((3, 3)), 0.3, "all", 3.0, 0.03
    )
    assert int(fallback[0]) == 0 and int(fallback[4]) == 0 and bool(fallback[5])


def test_fixed_shapes_reuse_risk_proposal_and_selection_compilations():
    config, model, _ = fixture()
    previous = jnp.zeros((3, 2), dtype=model.params.dtype)
    da = jnp.zeros((8, 3, 2), dtype=model.params.dtype)
    states = jnp.zeros((8, 3, 7), dtype=model.params.dtype)
    values = jnp.zeros(24, dtype=model.params.dtype)
    distance = jnp.zeros(8, dtype=model.params.dtype)
    risk = jnp.full((8, 3), 0.2, dtype=model.params.dtype)
    lambdas = jnp.zeros((4, 3), dtype=model.params.dtype)
    proposed = lmppi.propose_with_risk(
        previous, da, states, values, distance, risk, lambdas, config, "terminal"
    )
    proposed.block_until_ready()
    chosen = lmppi.select_risk_candidate(
        jnp.zeros((4, 3, 7)), jnp.zeros(4), jnp.full((4, 3), 0.2), 0.3,
        "terminal", config.boundary_limit, config.safe_set_relaxation,
    )
    jnp.asarray(chosen[0]).block_until_ready()
    before = (lmppi.propose_with_risk._cache_size(), lmppi.select_risk_candidate._cache_size())
    lmppi.propose_with_risk(
        previous, da, states, values, distance, risk + 0.1, lambdas, config,
        "terminal",
    ).block_until_ready()
    jnp.asarray(
        lmppi.select_risk_candidate(
            jnp.zeros((4, 3, 7)), jnp.zeros(4), jnp.full((4, 3), 0.4), 0.5,
            "terminal", config.boundary_limit, config.safe_set_relaxation,
        )[0]
    ).block_until_ready()
    assert before == (
        lmppi.propose_with_risk._cache_size(), lmppi.select_risk_candidate._cache_size()
    )


def test_risk_proposal_kernels_warm_only_when_active_before_proposals():
    config, model, _ = fixture()
    events = []
    planner = lmppi.LMPPI(config, model, SimpleNamespace(), risk_input_mode="state",
                          emit=lambda name, **fields: events.append((name, fields)))
    planner.precompile_hulls(lambda name, **fields: events.append((name, fields)))
    planner.precompile_hulls(lambda name, **fields: events.append((name, fields)))
    assert events == [] and planner.risk_lambdas is None
    planner.active_lambdas(False, True)
    planner.active_lambdas(False, True)
    assert events == [
        ("risk_constraint_warmed", {"candidates": 4, "mode": "terminal"})
    ]


def test_planner_scores_samples_and_weighted_candidates_with_matched_cutoff(monkeypatch):
    config, model, state = fixture()
    track = SimpleNamespace(get_s_search_range=lambda *args: np.arange(2))
    planner = lmppi.LMPPI(config, model, track, risk_input_mode="state_action")
    monkeypatch.setattr(lmppi, "to_frenet", lambda track, states, *args: states)
    monkeypatch.setattr(
        lmppi, "build_safe_geometry", lambda *args, **kwargs: SimpleNamespace(mask=np.ones(6))
    )
    planner.evaluate_hull = lambda states, hull: jnp.zeros(len(states))
    safe_set = SimpleNamespace(select=lambda *args: (np.zeros((6, 8)), np.zeros((2, 7))))
    monkeypatch.setattr(
        lmppi, "predict_value", lambda snapshot, states, *args: jnp.zeros(len(states))
    )
    queries = []
    candidate_actions = []
    prediction = [0.1]

    original_propose = lmppi.propose_with_risk

    def capture_proposal(*args):
        proposed = original_propose(*args)
        candidate_actions.append(np.asarray(proposed))
        return proposed

    monkeypatch.setattr(lmppi, "propose_with_risk", capture_proposal)

    def predict_risk(snapshot, states, actions=None):
        queries.append((np.asarray(states), np.asarray(actions)))
        return jnp.full(len(states), prediction[0])

    monkeypatch.setattr(lmppi, "predict_risk", predict_risk)
    value_snapshot = SimpleNamespace(input_mode="state", generation=1)
    risk_snapshot = SimpleNamespace(input_mode="state_action", generation=2)
    cutoff = {"status": "ready", "generation": 2, "threshold": 0.2}
    result = planner.update(
        state, np.asarray(state), value_snapshot, safe_set,
        risk_snapshot=risk_snapshot, risk_cutoff=cutoff, capture_debug=True,
    )
    assert result.metrics["risk_constraint_status"] == "active"
    assert result.metrics["risk_feasible_candidates"] == 4
    assert result.metrics["risk_score"] == pytest.approx(0.1)
    assert result.metrics["risk_fallback"] is False
    assert len(result.lambdas) == 3 and len(queries) == 2
    assert queries[0][0].shape == (24, 7) and queries[1][0].shape == (12, 7)
    sampled = np.concatenate(
        (result.debug["sample_actions"], result.debug["sample_endpoint_actions"][:, None]),
        axis=1,
    )
    np.testing.assert_allclose(
        queries[0][1].reshape(8, 3, 2), sampled[:, 1:] * model.action_scale
    )
    np.testing.assert_allclose(
        queries[1][1].reshape(4, 3, 2), candidate_actions[-1][:, 1:] * model.action_scale
    )
    assert result.debug["candidate_risk"].shape == (4, 3)
    assert result.debug["risk_mode"] == planner.config.risk_constraint.mode
    prediction[0] = 0.9
    fallback = planner.update(
        state, np.asarray(state), value_snapshot, safe_set,
        risk_snapshot=risk_snapshot, risk_cutoff=cutoff,
    )
    assert fallback.metrics["risk_constraint_status"] == "no_feasible_candidate"
    assert fallback.metrics["risk_feasible_candidates"] == 0
    assert fallback.metrics["risk_fallback"] is True
    with pytest.raises(ValueError, match="generation differs"):
        planner.update(
            state, np.asarray(state), value_snapshot, safe_set,
            risk_snapshot=risk_snapshot,
            risk_cutoff={**cutoff, "generation": 3},
        )
