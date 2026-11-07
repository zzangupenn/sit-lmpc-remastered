"""Constrained predictive-distribution fitting, not mean weights relabeled as Bayes."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from sit_lmpc.config import load_config
from sit_lmpc.config.schema import BeliefConfig
from sit_lmpc.data.trajectories import TrajectoryDataset
from sit_lmpc.dynamics.bank import STModelBank
from sit_lmpc.dynamics.distribution import STBankScenarioProvider
from sit_lmpc.dynamics.ensemble_fit import blend_loss_and_grad
from sit_lmpc.dynamics.model_bank import ModelBankBackend, energy_statistics
from test_dynamics_belief import model, trajectory
from test_ensemble_fit import make_learner, selection_for


def backend(dtype="float32", **settings):
    config = replace(BeliefConfig(), backend="model_bank", batch_size=8, **settings)
    return ModelBankBackend(STModelBank.from_model(model(0.4, dtype), (0.3, 0.4, 0.7, 1.0)),
                            config, horizon=3)


def test_energy_distances_match_direct_full_path_chord_embedding_and_masks():
    rng = np.random.default_rng(31)
    predictions = rng.normal(size=(4, 3, 3, 7))
    targets = rng.normal(size=(3, 3, 7))
    mask = np.array([[True] * 3, [True, False, False], [False] * 3])
    coefficients = np.array([0.6, 0.4, 0.])
    scales = np.arange(1, 8) / 10
    predictions[:, ~mask] = np.nan
    targets[~mask] = np.nan
    a, distances = energy_statistics(jnp.asarray(predictions), jnp.asarray(targets),
        jnp.asarray(mask), jnp.asarray(coefficients), jnp.asarray(scales), jnp.ones(4, dtype=bool))

    def embed(states):
        return np.stack([part for f in range(7) for part in
            ((np.sin(states[..., f]) / scales[f], np.cos(states[..., f]) / scales[f])
             if f in (2, 4, 6) else (states[..., f] / scales[f],))], axis=-1)

    expected_a, expected_d = np.zeros(4), np.zeros((4, 4))
    for row in range(2):
        members = embed(predictions[:, row, mask[row]]).reshape(4, -1)
        truth = embed(targets[row, mask[row]]).reshape(1, -1)
        normalization = np.sqrt(mask[row].sum() * 7)
        expected_a += coefficients[row] * cdist(members, truth)[:, 0] / normalization
        expected_d += coefficients[row] * cdist(members, members) / normalization
    np.testing.assert_allclose(a, expected_a, rtol=2e-6)
    np.testing.assert_allclose(distances, expected_d, rtol=2e-6)
    shifted = predictions.copy()
    shifted[..., 4] += 2 * np.pi
    wrapped = energy_statistics(jnp.asarray(shifted), jnp.asarray(targets), jnp.asarray(mask),
        jnp.asarray(coefficients), jnp.asarray(scales), jnp.ones(4, dtype=bool))
    np.testing.assert_allclose(wrapped[0], a, rtol=2e-6)
    np.testing.assert_allclose(wrapped[1], distances, rtol=2e-6)


@pytest.mark.parametrize("slack", [0., 1e-4, 0.1])
def test_distribution_improves_without_losing_mean_fit(monkeypatch, slack):
    fit = backend(mean_fit_slack=slack)
    _, selection = selection_for([trajectory(mu=0.5, count=12)], horizon=3)
    predictions = np.zeros((4, 2, 3, 7), dtype=np.float32)
    predictions[..., 0] = np.array([0, 1, 3, 4])[:, None, None]
    targets = np.zeros((2, 3, 7), dtype=np.float32)
    targets[..., 0] = 2
    cache = [(jnp.asarray(predictions), jnp.asarray(targets), jnp.ones((2, 3), dtype=bool),
              jnp.array([0.5, 0.5]))]
    fit.scales = jnp.ones(7)
    monkeypatch.setattr(fit, "_cache", lambda *args: cache)
    snapshot, metrics = fit.fit(selection, 1)
    assert snapshot is not None and metrics["mean_constraint_satisfied"]
    assert metrics["loss"] <= metrics["reference_loss"] + slack + metrics["constraint_tolerance"]
    assert metrics["energy_score"] < metrics["initial_energy_score"] - 0.01
    np.testing.assert_allclose(snapshot.weights, [0, 0.5, 0.5, 0], atol=2e-3)
    # Prediction and probability safety consume exactly the same published vector.
    provider = STBankScenarioProvider.from_snapshot(snapshot)
    assert provider.scenario_count == 4
    assert provider.probability_source == "fitted_predictive_distribution"
    np.testing.assert_array_equal(provider.probabilities, snapshot.weights)


def test_mean_error_constraint_actually_blocks_the_unconstrained_energy_optimum(monkeypatch):
    # For these two windows, regression puts 0.39 on member 1, while unconstrained
    # energy wants 0.475. The small MSE allowance must prevent that latter solution.
    fit = backend("float64", prior=(.5, .5, 0, 0), mean_fit_slack=1e-4,
                  fit_tolerance=1e-10)
    _, selection = selection_for([trajectory(mu=.5)], horizon=3)
    predictions = np.zeros((4, 2, 3, 7))
    predictions[1, 0, :, 0], predictions[1, 1, :, 0] = 1., 3.
    targets = np.zeros((2, 3, 7))
    targets[0, :, 0], targets[1, :, 0] = .9, 1.
    fit.scales = jnp.ones(7, dtype=jnp.float64)
    cached = [(jnp.asarray(predictions), jnp.asarray(targets),
               jnp.ones((2, 3), dtype=bool), jnp.array([.5, .5]))]
    monkeypatch.setattr(fit, "_cache", lambda *args: cached)
    snapshot, metrics = fit.fit(selection, 1)
    assert metrics["reference_weights"][1] == pytest.approx(.39, abs=1e-6)
    assert .39 < snapshot.weights[1] < .45
    assert metrics["mean_constraint_satisfied"]
    assert metrics["loss"] <= metrics["mean_loss_limit"] + metrics["constraint_tolerance"]
    assert metrics["loss"] > metrics["reference_loss"] + .5e-4
    assert metrics["energy_score"] < metrics["initial_energy_score"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_st_regression_reference_constraint_and_short_failure_windows(dtype):
    fit = backend(dtype, mean_fit_slack=1e-4)
    _, selection = selection_for([trajectory(mu=0.5, count=12, outcome="failure")], horizon=3)
    snapshot, metrics = fit.fit(selection, 1)
    assert snapshot is not None
    assert metrics["loss"] <= metrics["mean_loss_limit"] + metrics["constraint_tolerance"]
    assert metrics["energy_score"] <= metrics["initial_energy_score"] + 1e-10
    np.testing.assert_allclose(snapshot.weights.sum(), 1, atol=1e-7)
    assert np.min(snapshot.weights) >= 0
    cached = fit._cache(selection.windows, selection.weights)
    a, distances = fit._energy(cached)
    w = np.asarray(snapshot.weights, dtype=float)
    assert metrics["energy_score"] == pytest.approx(w @ a - 0.5 * w @ distances @ w)
    assert selection.windows.mask[-1].sum() == 1


def test_invalid_optimizer_result_retains_feasible_regression_reference(monkeypatch):
    from types import SimpleNamespace
    import sit_lmpc.dynamics.model_bank as module

    fit = backend()
    _, selected = selection_for([trajectory(mu=0.5)], 3)
    monkeypatch.setattr(module, "minimize", lambda *args, **kwargs:
        SimpleNamespace(x=np.array([np.nan] * 4), nit=1, success=False, message="test failure"))
    snapshot, metrics = fit.fit(selected, 1)
    assert snapshot is not None and metrics["used_reference"]
    assert not metrics["optimizer_converged"] and metrics["mean_constraint_satisfied"]


def test_excluded_support_and_single_model_are_preserved():
    fit = backend(prior=(0, 1, 0, 0))
    _, selected = selection_for([trajectory(mu=0.5)], 3)
    snapshot, metrics = fit.fit(selected, 1)
    np.testing.assert_array_equal(snapshot.weights, [0, 1, 0, 0])
    assert metrics["optimizer_iterations"] == 0


def test_exactly_four_coherent_models_and_event_probabilities_use_fitted_vector():
    from sit_lmpc.dynamics.bank import bank_rollout
    from sit_lmpc.planners.probabilistic import chance_probabilities

    fit = backend()
    _, selected = selection_for([trajectory(mu=.5)], 3)
    snapshot, _ = fit.fit(selected, 1)
    provider = STBankScenarioProvider.from_snapshot(snapshot)
    actions = jnp.zeros((2, 3, 2), dtype=fit.bank.params.dtype)
    state = jnp.asarray(selected.transitions.states[0], dtype=fit.bank.params.dtype)
    scenarios = provider.rollout(state, actions, key=jax.random.key(10))
    assert scenarios.states.shape == (2, 4, 3, 7)
    np.testing.assert_allclose(scenarios.states[0], bank_rollout(fit.bank, state, actions[0]))
    np.testing.assert_array_equal(provider.rollout(state, actions, key=jax.random.key(999)).states,
                                  scenarios.states)
    frenet = jnp.zeros_like(scenarios.states).at[:, 1, :2, 1].set(5.)
    hull = jnp.zeros((2, 4), dtype=frenet.dtype).at[:, 2].set(.2)
    chance = chance_probabilities(frenet, scenarios.valid, scenarios.weights, hull, 0., 4., .03)
    np.testing.assert_allclose(chance["boundary"], np.asarray(snapshot.weights)[1])
    np.testing.assert_allclose(chance["terminal_hull"], np.asarray(snapshot.weights)[2])


def test_nonfinite_or_mismatched_windows_do_not_publish():
    fit = backend()
    _, selected = selection_for([trajectory(mu=0.5)], 3)
    with pytest.raises(ValueError, match="recorded rollout windows"):
        fit.fit(replace(selected, windows=None), 1)
    bad = selected.windows.targets.copy()
    bad[0, 0, 0] = np.nan
    snapshot, metrics = fit.fit(replace(selected, windows=replace(selected.windows, targets=bad)), 1)
    assert snapshot is None and metrics["status"] == "ensemble_mismatch"


def test_fixed_shapes_reuse_rollout_mean_and_energy_compilation():
    from sit_lmpc.dynamics.ensemble_fit import window_predictions

    for kernel in (window_predictions, blend_loss_and_grad, energy_statistics):
        kernel.clear_cache()
    fit = backend()
    for count in (12, 19):
        _, selected = selection_for([trajectory(mu=0.5, count=count)], 3)
        assert fit.fit(selected, count)[0] is not None
    assert all(kernel._cache_size() == 1
               for kernel in (window_predictions, blend_loss_and_grad, energy_statistics))


def test_restore_and_changed_slack_or_legacy_method_refit(tmp_path):
    config = load_config("tests/fixtures/presets/st-model-bank.yaml")
    dataset = TrajectoryDataset(tmp_path, config, 12)
    record = trajectory(mu=0.4, count=20)
    dataset.add(record)
    fit = make_learner(tmp_path, horizon=3, backend_name="model_bank")
    fit.observe(record)
    assert fit.update()
    original = np.asarray(fit.snapshot.weights).copy()
    restored = make_learner(tmp_path, horizon=3, backend_name="model_bank")
    restored.restore(tmp_path, .02)
    assert restored.generation == 1
    np.testing.assert_array_equal(restored.snapshot.weights, original)
    changed = make_learner(tmp_path, horizon=3, backend_name="model_bank", mean_fit_slack=.01)
    changed.restore(tmp_path, .02)
    assert changed.generation == 2
    state_path = tmp_path / "dynamics_belief.json"
    state = json.loads(state_path.read_text())
    state.pop("method")  # Old model_bank used likelihood, even with the same backend name.
    state_path.write_text(json.dumps(state))
    migrated = make_learner(tmp_path, horizon=3, backend_name="model_bank", mean_fit_slack=.01)
    migrated.restore(tmp_path, .02)
    assert migrated.generation == 3
    assert migrated.last_fit["method"] == "mean_constrained_energy_v1"


@pytest.mark.parametrize("slack", [-1, True, float("nan"), float("inf")])
def test_slack_validation(slack):
    with pytest.raises(ValueError, match="mean_fit_slack"):
        load_config(overrides={"planner": {"belief": {"mean_fit_slack": slack}}})


def test_presets_alias_and_legacy_likelihood_are_explicit():
    fitted = load_config("tests/fixtures/presets/st-model-bank.yaml")
    regression = load_config("tests/fixtures/presets/st-ensemble-fit.yaml")
    assert fitted.planner.belief.backend == "model_bank"
    assert regression.planner.belief.backend == "ensemble_fit"
    assert BeliefConfig(backend="fit_ensemble").backend == "ensemble_fit"
    assert fitted.planner.model_bank.frictions == (.3, .4, .7, 1.)
    assert fitted.simulation.friction == .5
    assert fitted.planner.probabilistic_safety.enabled and not fitted.risk.enabled
    assert load_config(overrides={"planner": {"belief": {"backend": "likelihood"}}})
    hash(fitted.planner)
