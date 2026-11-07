"""Stage 6 probability scenarios, thresholds, gates, and disabled-term work."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from sit_lmpc.config.schema import (
    ExperimentConfig,
    ProbabilisticSafetyConfig,
    RiskConstraintConfig,
    load_config,
)
from sit_lmpc.config.resume import resume_overrides
from sit_lmpc.dynamics.bank import STModelBank, bank_rollout
from sit_lmpc.dynamics.belief import BeliefSnapshot
from sit_lmpc.dynamics.distribution import STBankScenarioProvider, TrajectoryScenarios
from sit_lmpc.dynamics.ensemble import STEnsemble
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.planners import lmppi
from sit_lmpc.planners.probabilistic import (
    SafetyStop,
    chance_probabilities,
    lambda_table,
    select_constrained_candidate,
)


def example(dtype=jnp.float32):
    states = jnp.zeros((2, 3, 4, 7), dtype=dtype)
    # Two boundary-violating steps in one scenario must count only once.
    states = states.at[0, 1, 1:3, 1].set(5).at[1, 2, 0, 1].set(-5)
    p = jnp.asarray([0.6, 0.3, 0.1], dtype=dtype)
    valid = jnp.ones(states.shape[:-1], dtype=bool)
    hull = jnp.asarray([[0, 0, 0.2], [0, 0.2, 0.2]], dtype=dtype)
    return states, valid, p, hull


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_exact_events_are_weighted_per_trajectory_not_steps_or_mean(dtype):
    states, valid, p, hull = example(dtype)
    result = chance_probabilities(states, valid, p, hull, 0.0, 4.0, 0.03)
    np.testing.assert_allclose(result["boundary"], [0.3, 0.1], atol=1e-7)
    np.testing.assert_allclose(result["terminal_hull"], [0.1, 0.4], atol=1e-7)
    assert np.asarray(result["available"]).all()
    assert (
        np.abs(
            np.sum(np.asarray(states) * np.asarray(p)[None, :, None, None], axis=1)[..., 1]
        ).max()
        < 4
    )


def test_missing_endpoints_keep_failed_mass_and_zero_mass_nan_is_ignored():
    states, valid, _, hull = example()
    states = states.at[:, 2].set(jnp.nan)
    p = jnp.asarray([0.75, 0.25, 0.0])
    valid = valid.at[:, 1, 2:].set(False)
    hull = hull.at[:, 2].set(jnp.nan)
    result = chance_probabilities(states, valid, p, hull, 0.0, 4.0, 0.03)
    np.testing.assert_allclose(result["boundary"], [0.25, 0.25])
    np.testing.assert_allclose(result["terminal_hull"], [0.25, 0.25])
    assert np.asarray(result["available"]).all()
    # A stopped branch must not restart in the same trajectory.
    invalid = chance_probabilities(states, valid.at[0, 1, 3].set(True), p, hull, 0.0, 4.0, 0.03)
    assert not bool(invalid["available"][0])


@pytest.mark.parametrize(
    "problem", ["nan_state", "nan_hull", "negative", "unnormalized", "unknown_bound"]
)
def test_invalid_supported_mass_cannot_be_renormalized_into_safety(problem):
    states, valid, p, hull = example()
    error = 0.0
    if problem == "nan_state":
        states = states.at[:, 1, 1].set(jnp.nan)
    elif problem == "nan_hull":
        hull = hull.at[:, 1].set(jnp.nan)
    elif problem == "negative":
        p = jnp.asarray([0.8, 0.3, -0.1])
    elif problem == "unnormalized":
        p = p * 0.9
    else:
        error = jnp.nan
    result = chance_probabilities(states, valid, p, hull, error, 4.0, 0.03)
    assert not np.asarray(result["available"]).any()


def test_declared_sampling_error_changes_gate_not_raw_probability():
    states, valid, p, hull = example()
    result = chance_probabilities(states, valid, p, hull, 0.1, 4.0, 0.03)
    np.testing.assert_allclose(result["boundary"], [0.3, 0.1])
    np.testing.assert_allclose(result["boundary_upper"], [0.4, 0.2])
    candidate = jnp.zeros((2, 4, 7))
    selected = select_constrained_candidate(
        candidate, jnp.zeros(2), result, None, None, 0.3, 0.5, "all", 4.0, 0.03
    )
    assert int(selected[0]) == 1 and int(selected[3]) == 1


def test_empirical_64_sample_gate_allows_three_violations_not_four():
    states = jnp.zeros((2, 64, 3, 7))
    states = states.at[0, :3, 0, 1].set(5).at[1, :4, 0, 1].set(5)
    chance = chance_probabilities(states, jnp.ones(states.shape[:-1], bool),
                                 jnp.full(64, 1 / 64), jnp.zeros((2, 64)), 0.0, 4.0, 0.03)
    np.testing.assert_array_equal(chance["boundary"], [3 / 64, 4 / 64])
    selected = select_constrained_candidate(jnp.zeros((2, 3, 7)), jnp.zeros(2),
        chance, None, None, 0.05, 0.05, "terminal", 4.0, 0.03)
    assert int(selected[0]) == 0 and int(selected[3]) == 1 and not bool(selected[4])


@pytest.mark.parametrize("mode,index,count", [("all", 1, 1), ("terminal", 0, 2)])
def test_joint_intersection_and_inclusive_thresholds(mode, index, count):
    states = jnp.zeros((3, 3, 7))
    chance = {
        "available": jnp.ones(3, dtype=bool),
        "boundary_upper": jnp.asarray([0.2, 0.2, 0.21]),
        "terminal_hull_upper": jnp.asarray([0.1, 0.1, 0.0]),
    }
    risk = jnp.asarray([[0.9, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
    result = select_constrained_candidate(
        states, jnp.zeros(3), chance, risk, 0.3, 0.2, 0.1, mode, 4.0, 0.03
    )
    assert int(result[0]) == index and int(result[3]) == count and not bool(result[4])


@pytest.mark.parametrize("action_value", [False, True])
def test_no_feasible_fallback_matches_sit_geometric_score_and_first_tie(action_value):
    states = jnp.zeros((6, 3, 7)).at[0, 0, 1].set(4.1).at[4, 0, 0].set(jnp.nan)
    distances = jnp.asarray([0.0, 0.9, 0.1, 0.1, 0.0, jnp.nan])
    chance = {
        "available": jnp.ones(6, dtype=bool),
        "boundary_upper": jnp.asarray([0.9, 0.8, 0.7, 0.1, 0.0, 0.0]),
        "terminal_hull_upper": jnp.ones(6),
    }
    # Neither the lowest chance probability, largest lambda index, nor highest
    # action value replaces SIT's minimum violation/first-equal-score rule.
    terminal_values = jnp.asarray([0., 0., -100., 100., 0., 0.]) if action_value else None
    result = select_constrained_candidate(
        states, distances, chance, None, None, 0.05, 0.05,
        "terminal", 4.0, 0.03, terminal_values,
    )
    assert int(result[0]) == 2 and int(result[3]) == 0 and bool(result[4])
    assert float(result[1]) == pytest.approx(0.1) and float(result[2]) == 0.0


@pytest.mark.parametrize("preset", [None, "tests/fixtures/presets/bnn.yaml", "configs/bnn-posterior.yaml",
                                   "tests/fixtures/presets/bnn96-chance.yaml", "tests/fixtures/presets/st-model-bank.yaml",
                                   "tests/fixtures/presets/st-ensemble-fit.yaml"])
def test_presets_use_sit_fallback_by_default_but_keep_explicit_stop(preset):
    assert load_config(preset).planner.probabilistic_safety.fallback == "baseline"
    assert ProbabilisticSafetyConfig().fallback == "baseline"
    assert load_config(preset, overrides={"planner": {"probabilistic_safety": {
        "fallback": "stop"}}}).planner.probabilistic_safety.fallback == "stop"


def test_bank_adapter_preserves_fixed_friction_and_separate_posterior():
    base = ExperimentConfig()
    _, fixed = build_vehicle_dynamics(base)
    bank = STModelBank.from_model(fixed, (0.3, 0.6))
    coefficients = jnp.asarray([0.95, 0.05], dtype=fixed.params.dtype)
    posterior = jnp.asarray([0.2, 0.8], dtype=fixed.params.dtype)
    snapshot = BeliefSnapshot(bank, coefficients, 7, "ensemble_fit", posterior)
    provider = STBankScenarioProvider.from_snapshot(snapshot)
    state = jnp.asarray([0, 0, 0.1, 10, 0, 0, 0], dtype=fixed.params.dtype)
    actions = jnp.full((3, 4, 2), 0.2, dtype=fixed.params.dtype)
    result = provider.rollout(state, actions, key=jax.random.key(12))
    np.testing.assert_array_equal(result.weights, posterior)
    np.testing.assert_allclose(result.states[1], bank_rollout(bank, state, actions[1]))
    assert result.valid.shape == (3, 2, 4) and result.valid.all()
    assert provider.metadata["belief_generation"] == 7
    assert provider.can_reuse(STEnsemble(bank, coefficients, "ensemble_coefficients"))
    assert STBankScenarioProvider.from_snapshot(replace(snapshot, posterior_weights=None)) is None


class MockProvider:
    """Coherent GP/flow-like full-path draws; explicit test probability envelope."""

    scenario_count = 2
    metadata = {
        "method": "mock_posterior",
        "sample_count": 2,
        "probability_source": "posterior_draws",
        "belief_generation": 3,
        "confidence": 0.95,
    }

    def __init__(self, y=0.0, error=0.0):
        self.y, self.error, self.calls = y, error, []

    def rollout(self, state, controls, *, key, scan=True):
        self.calls.append((np.asarray(jax.random.key_data(key)), np.asarray(controls)))
        n, horizon = controls.shape[:2]
        states = jnp.broadcast_to(state, (n, 2, horizon, 7))
        states = states.at[:, 0, :, 1].set(0).at[:, 1, :, 1].set(self.y)
        return TrajectoryScenarios(
            states,
            jnp.asarray([0.8, 0.2], dtype=states.dtype),
            jnp.ones(states.shape[:-1], dtype=bool),
            self.error,
        )


def setup(monkeypatch, chance, risk, evaluation="predictor", provider=None, fallback="stop"):
    base = ExperimentConfig()
    config = replace(
        base.planner,
        horizon=3,
        samples=8,
        lambda_count=2,
        hull_precompile=False,
        probabilistic_safety=ProbabilisticSafetyConfig(
            enabled=chance,
            boundary_threshold=0.25,
            terminal_hull_threshold=0.25,
            lambda_count=2,
            max_candidates=3,
            fallback=fallback,
        ),
        risk_constraint=RiskConstraintConfig(
            enabled=risk,
            evaluation=evaluation,
            lambda_count=2,
            max_candidates=3,
        ),
    )
    _, model = build_vehicle_dynamics(base)
    state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=model.params.dtype)
    track = SimpleNamespace(get_s_search_range=lambda *args: np.arange(2))
    planner = lmppi.LMPPI(config, model, track, scenario_provider=provider)
    monkeypatch.setattr(lmppi, "to_frenet", lambda track, states, *args: states)
    monkeypatch.setattr(
        lmppi, "build_safe_geometry", lambda *args, **kwargs: SimpleNamespace(mask=np.ones(6))
    )
    planner.evaluate_hull = lambda states, hull: jnp.zeros(len(states))
    monkeypatch.setattr(
        lmppi,
        "predict_value",
        lambda snapshot, states, *args: jnp.zeros(len(states), dtype=states.dtype),
    )
    safe = SimpleNamespace(select=lambda *args: (np.zeros((6, 8)), np.zeros((2, 7))))
    value = SimpleNamespace(input_mode="state", generation=1)
    risk_snapshot = SimpleNamespace(input_mode="state", generation=2)
    cutoff = {"status": "ready", "generation": 2, "threshold": 0.5}
    return planner, state, safe, value, risk_snapshot, cutoff


@pytest.mark.parametrize(
    "chance,risk,dimensions",
    [(False, False, 2), (True, False, 4), (False, True, 3), (True, True, 5)],
)
def test_four_toggles_remove_unused_tables_inference_and_scenarios(
    monkeypatch, chance, risk, dimensions
):
    provider = MockProvider()
    planner, state, safe, value, snapshot, cutoff = setup(
        monkeypatch, chance, risk, provider=provider
    )
    calls = []

    def predict(snapshot, states, actions=None):
        if not risk:
            pytest.fail("disabled risk invoked control inference")
        calls.append(len(states))
        return jnp.full(len(states), 0.2)

    monkeypatch.setattr(lmppi, "predict_risk", predict)
    planner.precompile_hulls()
    assert planner._lambda_tables.keys() == {(False, False)} and planner.risk_lambdas is None
    result = planner.update(
        state, state, value, safe, risk_snapshot=snapshot, risk_cutoff=cutoff if risk else None
    )
    assert len(result.lambdas) == dimensions
    assert len(result.metrics["lambda_terms"]) == dimensions
    assert len(provider.calls) == (2 if chance else 0)
    assert len(calls) == (2 if risk else 0)
    assert planner.actions.shape == (3, 2)
    if chance:
        assert result.metrics["probabilistic_safety_status"] == "active"
        np.testing.assert_array_equal(provider.calls[0][0], provider.calls[1][0])
        assert len(result.lambdas) == 4 + int(risk)
    if not risk:
        assert planner.risk_lambdas is None and not planner._risk_warmed


def test_no_ready_cutoff_has_no_risk_axis_or_endpoint_even_with_snapshot(monkeypatch):
    planner, state, safe, value, snapshot, _ = setup(monkeypatch, False, True)
    snapshot.input_mode = "state_action"
    monkeypatch.setattr(
        lmppi, "predict_risk", lambda *args: pytest.fail("unavailable risk inference")
    )
    result = planner.update(state, state, value, safe, risk_snapshot=snapshot)
    assert result.metrics["risk_constraint_status"] == "unavailable"
    assert len(result.lambdas) == 2 and planner.actions.shape == (3, 2)
    assert planner.risk_lambdas is None and not planner._risk_warmed


def test_distribution_risk_scores_member_states_and_next_actions(monkeypatch):
    provider = MockProvider(y=2.0)
    planner, state, safe, value, snapshot, cutoff = setup(
        monkeypatch, True, True, "distribution", provider
    )
    snapshot.input_mode = "state_action"
    calls = []

    def predict(snapshot, states, actions=None):
        calls.append((np.asarray(states), np.asarray(actions)))
        # q(mean_state) differs from mean q; prove queries are branch states.
        return jnp.square(states[:, 1]) * 0.1

    monkeypatch.setattr(lmppi, "predict_risk", predict)
    result = planner.update(
        state, state, value, safe, risk_snapshot=snapshot, risk_cutoff=cutoff, capture_debug=True
    )
    assert result.metrics["risk_score"] == pytest.approx(0.08)
    assert result.metrics["risk_evaluation"] == "distribution"
    assert calls[0][0].shape == (8 * 2 * 3, 7)
    sampled = np.concatenate(
        (result.debug["sample_actions"], result.debug["sample_endpoint_actions"][:, None]), axis=1
    )
    expected = np.broadcast_to(sampled[:, None, 1:] * planner.model.action_scale, (8, 2, 3, 2))
    np.testing.assert_allclose(calls[0][1].reshape(8, 2, 3, 2), expected)
    np.testing.assert_allclose(result.debug["risk_probabilities"], 0.08)


@pytest.mark.parametrize(
    "failure", ["missing_distribution", "unknown_sampling_error", "no_feasible_candidate"]
)
def test_default_failure_policy_stops_before_modifying_action_horizon(monkeypatch, failure):
    provider = (
        None
        if failure == "missing_distribution"
        else MockProvider(
            y=5.0 if failure == "no_feasible_candidate" else 0.0,
            error=None if failure == "unknown_sampling_error" else 0.0,
        )
    )
    planner, state, safe, value, _, _ = setup(monkeypatch, True, False, provider=provider)
    if failure == "no_feasible_candidate":
        planner.config = replace(
            planner.config,
            probabilistic_safety=replace(
                planner.config.probabilistic_safety, boundary_threshold=0.1
            ),
        )
    original = np.asarray(planner.actions).copy()
    with pytest.raises(SafetyStop) as stopped:
        planner.update(state, state, value, safe)
    assert stopped.value.reason == (
        "no_feasible_candidate" if failure == "no_feasible_candidate" else "unavailable"
    )
    np.testing.assert_array_equal(planner.actions, original)
    if failure != "no_feasible_candidate":
        assert planner._lambda_tables.keys() == {(False, False)}


def test_sit_fallback_and_candidate_reroll_not_sample_gate(monkeypatch):
    class Changing(MockProvider):
        def rollout(self, state, controls, **kwargs):
            # Original samples safe, weighted candidates unsafe: check reroll.
            self.y = 0.0 if len(controls) == 8 else 5.0
            return super().rollout(state, controls, **kwargs)

    provider = Changing()
    planner, state, safe, value, _, _ = setup(
        monkeypatch, True, False, provider=provider, fallback="baseline"
    )
    planner.config = replace(
        planner.config,
        probabilistic_safety=replace(planner.config.probabilistic_safety, boundary_threshold=0.1),
    )
    result = planner.update(state, state, value, safe)
    assert result.metrics["chance_fallback"] is True
    assert result.metrics["probabilistic_safety_status"] == "no_feasible_candidate"
    assert result.metrics["chance_fallback_policy"] == "sit_conservative"
    assert result.metrics["boundary_probability"] == pytest.approx(0.2)
    assert result.metrics["boundary_probability_upper"] == pytest.approx(0.2)
    assert result.metrics["boundary_probability_threshold"] == pytest.approx(0.1)
    assert result.metrics["terminal_hull_probability"] == 0.0
    assert result.metrics["chance_feasible_candidates"] == 0
    assert np.isfinite(result.action).all()


def test_sit_fallback_keeps_unavailable_probabilities_explicit(monkeypatch):
    planner, state, safe, value, _, _ = setup(
        monkeypatch, True, False, fallback="baseline"
    )
    result = planner.update(state, state, value, safe)
    assert result.metrics["chance_fallback_policy"] == "sit_conservative"
    assert result.metrics["chance_unavailable_reason"] == "dynamics_distribution_unavailable"
    assert "boundary_probability" not in result.metrics
    assert "terminal_hull_probability" not in result.metrics
    assert len(result.lambdas) == 2


def test_risk_distribution_without_probability_safety_or_provider_is_explicitly_unavailable(
    monkeypatch,
):
    planner, state, safe, value, snapshot, cutoff = setup(monkeypatch, False, True, "distribution")
    monkeypatch.setattr(
        lmppi, "predict_risk", lambda *args: pytest.fail("unavailable risk inference")
    )
    result = planner.update(state, state, value, safe, risk_snapshot=snapshot, risk_cutoff=cutoff)
    assert result.metrics["risk_unavailable_reason"] == "dynamics_distribution_unavailable"
    assert len(result.lambdas) == 2 and planner._lambda_tables.keys() == {(False, False)}


@pytest.mark.parametrize(
    "key,value",
    [
        ("boundary_threshold", -0.1),
        ("terminal_hull_threshold", 1.1),
        ("boundary_threshold", True),
        ("terminal_hull_threshold", float("nan")),
        ("lambda_count", 0),
        ("max_candidates", 0),
        ("lambda_max", 0),
        ("fallback", "silent"),
    ],
)
def test_invalid_probability_config_is_rejected(key, value):
    base = ExperimentConfig()
    config = replace(
        base,
        planner=replace(
            base.planner,
            probabilistic_safety=replace(base.planner.probabilistic_safety, **{key: value}),
        ),
    )
    with pytest.raises(ValueError, match="probabilistic_safety"):
        config.validate()


def test_hashable_yaml_and_resume_can_change_thresholds_and_toggles(tmp_path):
    base = ExperimentConfig()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base.as_dict()))
    changes = {
        "planner": {
            "probabilistic_safety": {"enabled": True, "boundary_threshold": 0.2},
            "risk_constraint": {"evaluation": "distribution", "enabled": True},
        }
    }
    loaded = load_config(path, overrides=resume_overrides(changes))
    assert loaded.planner.probabilistic_safety.boundary_threshold == 0.2
    assert loaded.planner.probabilistic_safety.terminal_hull_threshold == 0.05
    assert loaded.planner.risk_constraint.evaluation == "distribution"
    assert isinstance(hash(loaded.planner), int)


def test_active_lambda_design_is_bounded_and_has_no_inactive_columns():
    base = ExperimentConfig().planner
    base_table = jnp.zeros((256, 2))
    config = replace(
        base,
        probabilistic_safety=replace(base.probabilistic_safety, max_candidates=64),
        risk_constraint=replace(base.risk_constraint, max_candidates=32),
    )
    assert lambda_table(base_table, config, False, False) is base_table
    for chance, risk, shape in (
        (True, False, (64, 4)),
        (False, True, (32, 3)),
        (True, True, (32, 5)),
    ):
        result = lambda_table(base_table, config, chance, risk)
        assert result.shape == shape and np.asarray(result).max() < 20
        np.testing.assert_array_equal(result[0], 0)
        assert np.asarray(result)[:, 2:].max(axis=0).min() > 0


def test_disabled_optional_terms_preserve_rng_baseline_and_skip_distribution(monkeypatch):
    class Unused(MockProvider):
        def rollout(self, *args, **kwargs):
            pytest.fail("disabled optional terms requested scenarios")

    baseline, state, safe, value, _, _ = setup(monkeypatch, False, False)
    disabled, _, _, _, snapshot, _ = setup(monkeypatch, False, False, provider=Unused())
    snapshot.input_mode = "state_action"
    monkeypatch.setattr(lmppi, "predict_risk", lambda *args: pytest.fail("disabled risk query"))
    for _ in range(3):
        first = baseline.update(state, state, value, safe)
        second = disabled.update(state, state, value, safe, risk_snapshot=snapshot)
        np.testing.assert_array_equal(first.action, second.action)
        np.testing.assert_array_equal(first.trajectory, second.trajectory)
        np.testing.assert_array_equal(baseline.actions, disabled.actions)
        assert len(second.lambdas) == 2 and second.risk is None


@pytest.mark.parametrize("method", ["mock_gp", "mock_flow"])
def test_generic_temporally_coherent_posterior_draws_share_latent_across_candidates(
    monkeypatch, method
):
    class Coherent(MockProvider):
        scenario_count = 8
        metadata = {
            "method": method,
            "sample_count": 8,
            "probability_source": "iid_posterior_draws",
            "belief_generation": 9,
            "confidence": 0.95,
        }

        def rollout(self, state, controls, *, key, scan=True):
            controls = jnp.asarray(controls, dtype=state.dtype)
            latent = jax.random.normal(key, (8,), dtype=state.dtype) * 0.01
            self.calls.append(np.asarray(latent))
            n, horizon = controls.shape[:2]
            states = jnp.broadcast_to(state, (n, 8, horizon, 7))
            times = jnp.arange(1, horizon + 1, dtype=state.dtype)
            y = jnp.cumsum(controls[..., 0], axis=1)[:, None] + latent[None, :, None] * times
            states = states.at[..., 1].set(y)
            return TrajectoryScenarios(
                states,
                jnp.full(8, 1 / 8, dtype=state.dtype),
                jnp.ones(states.shape[:-1], dtype=bool),
                jnp.asarray(0.1, dtype=state.dtype),
            )

    provider = Coherent()
    planner, state, safe, value, _, _ = setup(monkeypatch, True, False, provider=provider)
    result = planner.update(state, state, value, safe)
    assert result.metrics["method"] == method
    assert result.metrics["chance_probability_error_bound"] == pytest.approx(0.1)
    np.testing.assert_array_equal(provider.calls[0], provider.calls[1])
