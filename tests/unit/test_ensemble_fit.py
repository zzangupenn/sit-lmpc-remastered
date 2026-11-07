"""Prediction fitting uses real windows and never relabels coefficients as probabilities."""

from dataclasses import replace
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.config.schema import BeliefConfig
from sit_lmpc.data.dynamics import DynamicsReplay, trajectory_transitions
from sit_lmpc.data.trajectories import TrajectoryDataset
from sit_lmpc.dynamics.bank import STModelBank, bank_step
from sit_lmpc.dynamics.belief import LikelihoodBackend
from sit_lmpc.dynamics.ensemble import weighted_states
from sit_lmpc.dynamics.ensemble_fit import (
    EnsembleFitBackend,
    blend_loss_and_grad,
    window_predictions,
)
from sit_lmpc.experiments.randomness import ExperimentRandomness
from sit_lmpc.training.dynamics import DynamicsBeliefLearner
from test_dynamics_belief import model, trajectory


def selection_for(records, horizon=10):
    replay = DynamicsReplay()
    for record in records:
        replay.add(record.attempt_id, record.outcome, trajectory_transitions(record, 0.02))
    selection = replay.select(np.random.default_rng(7), 16, 0.8)
    return replay, replace(selection, windows=replay.windows(selection, horizon))


def backend(dtype="float32", **settings):
    config = replace(BeliefConfig(), backend="ensemble_fit", batch_size=8, **settings)
    return EnsembleFitBackend(STModelBank.from_model(model(0.7, dtype), (0.3, 0.6)), config, 10)


def make_learner(directory, *, horizon=10, backend_name="ensemble_fit", **settings):
    config = replace(
        BeliefConfig(),
        enabled=True,
        backend=backend_name,
        predictor="ensemble",
        batch_size=8,
        history_samples=16,
        **settings,
    )
    return DynamicsBeliefLearner(
        config,
        model(0.7),
        (0.3, 0.6),
        directory,
        ExperimentRandomness(19).fork("dynamics_belief"),
        prediction_horizon=horizon,
    )


def test_windows_follow_full_source_not_sparse_history_and_keep_short_failures():
    records = [
        trajectory(i, outcome, count=12, mu=0.4)
        for i, outcome in enumerate(("success", "failure", "success", "failure"), 1)
    ]
    replay, selection = selection_for(records, 10)
    assert selection.recent_count == 24 and selection.history_count == 16
    windows = selection.windows
    by_id = {record.attempt_id: record for record in records}
    for row, (attempt_id, step_id) in enumerate(selection.transitions.ids):
        record = by_id[int(attempt_id)]
        valid = min(10, len(record.controls) - int(step_id))
        assert windows.mask[row].sum() == valid
        np.testing.assert_array_equal(
            windows.controls[row, :valid], record.controls[step_id : step_id + valid]
        )
        np.testing.assert_array_equal(
            windows.targets[row, :valid], record.cartesian[step_id + 1 : step_id + valid + 1]
        )
    assert selection.weights[:24].sum() == pytest.approx(0.8)
    assert selection.weights[24:].sum() == pytest.approx(0.2)
    np.testing.assert_array_equal(selection.transitions.ids, replay.last_selection)
    row = np.flatnonzero((selection.transitions.ids == [4, 11]).all(axis=1))[0]
    assert windows.mask[row].tolist() == [True] + [False] * 9


def test_windows_stop_at_contacts_and_discontinuous_reset_or_init_segments():
    record = trajectory(count=12, mu=0.4)
    contacts = record.contacts.copy()
    contacts[3] = True
    replay, selected = selection_for([replace(record, contacts=contacts)], 10)
    assert selected.windows.mask[0].sum() == 3
    assert selected.windows.mask[3].sum() == 8  # Next valid start is physical step 4.
    batch = replay.episodes[1][1]
    states = batch.states.copy()
    states[5, 0] += 100  # IDs alone must not join a reset/disjoint warm-up segment.
    replay.episodes[1] = ("success", replace(batch, states=states))
    assert replay.windows(selected, 10).mask[4].sum() == 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_joint_horizon_fit_improves_off_grid_predictions_and_keeps_posterior_separate(dtype):
    _, selected = selection_for([trajectory(mu=0.4, count=32, steering=0.2)], 10)
    fit = backend(dtype, evidence_strength=32)
    posterior, _ = LikelihoodBackend(fit.bank, fit.config).fit(selected, 1)
    cached = fit._cache(selected.windows, selected.weights)
    before = fit._metrics(posterior.weights, cached)
    snapshot, metrics = fit.fit(selected, 1)
    assert snapshot is not None and metrics["status"] == "ready"
    assert metrics["ensemble_position_rmse"] < before["ensemble_position_rmse"] * 0.3
    assert metrics["loss"] <= metrics["initial_loss"]
    assert snapshot.weight_kind == "ensemble_coefficients"
    assert snapshot.backend == "ensemble_fit"
    assert np.min(snapshot.weights) > 0.05 and snapshot.weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(snapshot.categorical_weights, posterior.weights, atol=1e-6)
    assert not np.allclose(snapshot.weights, snapshot.categorical_weights, atol=0.1)


def test_fitter_can_leave_a_previous_one_hot_solution():
    _, selected = selection_for([trajectory(mu=0.4, count=20, steering=0.2)])
    fit = backend()
    fit.weights = np.asarray([1.0, 0.0])
    snapshot, metrics = fit.fit(selected, 1)
    assert snapshot is not None and snapshot.weights[1] > 0.1
    assert metrics["loss"] < metrics["initial_loss"] * 0.1


def test_ensemble_fit_can_succeed_when_no_individual_model_passes_likelihood_guard():
    _, selected = selection_for([trajectory(mu=0.4, count=32, steering=0.2)])
    baseline = backend(max_standardized_rmse=50)
    _, metrics = baseline.fit(selected, 1)
    member_rmse = baseline.statistics(selected.transitions, selected.weights)[
        "standardized_rmse"
    ].min()
    assert metrics["ensemble_standardized_rmse"] < member_rmse
    limit = (metrics["ensemble_standardized_rmse"] + member_rmse) / 2
    fit = backend(max_standardized_rmse=float(limit))
    snapshot, metrics = fit.fit(selected, 1)
    assert snapshot is not None and snapshot.categorical_weights is None
    assert metrics["posterior_status"] == "all_models_mismatch"
    assert snapshot.weight_kind == "ensemble_coefficients"


def test_score_is_actual_ensemble_prediction_error_before_training(tmp_path):
    fit = make_learner(tmp_path)
    record = trajectory(mu=0.4, count=12)
    diagnostic = fit.observe(record)
    batch = fit.replay.episodes[1][1]
    predictions = jax.vmap(
        lambda x, u: bank_step(fit.snapshot.bank, x, u / fit.snapshot.bank.action_scale)
    )(jnp.asarray(batch.states), jnp.asarray(batch.controls, dtype=jnp.float32))
    mean = np.asarray(weighted_states(jnp.swapaxes(predictions, 0, 1), fit.snapshot.weights))
    residual = batch.next_states - mean
    for field in (2, 4, 6):
        residual[:, field] = np.arctan2(np.sin(residual[:, field]), np.cos(residual[:, field]))
    np.testing.assert_allclose(
        diagnostic["ensemble_component_rmse"], np.sqrt(np.mean(residual**2, axis=0)), atol=3e-6
    )
    assert diagnostic["ensemble_position_rmse"] < np.linalg.norm(
        diagnostic["weighted_component_rmse"][:2]
    )


def test_wrapped_angle_and_nonfinite_masked_padding_have_same_fit():
    _, selected = selection_for([trajectory(mu=0.4, count=12)])
    fit = backend()
    first, metrics = fit.fit(selected, 1)
    targets = selected.windows.targets.copy()
    targets[..., 4] += 2 * np.pi
    targets[~selected.windows.mask] = np.nan
    other = replace(selected, windows=replace(selected.windows, targets=targets))
    second, duplicate = backend().fit(other, 1)
    np.testing.assert_allclose(first.weights, second.weights, atol=1e-4)
    assert duplicate["loss"] == pytest.approx(metrics["loss"], abs=1e-5)


def test_padded_training_kernels_do_not_recompile_for_new_replay_sizes():
    window_predictions.clear_cache()
    blend_loss_and_grad.clear_cache()
    fit = backend()
    for count in (12, 19):
        _, selected = selection_for([trajectory(mu=0.4, count=count)])
        assert fit.fit(selected, count)[0] is not None
    assert window_predictions._cache_size() == blend_loss_and_grad._cache_size() == 1


def test_resume_restores_coefficients_posterior_and_refits_backend_or_horizon_changes(tmp_path):
    config = load_config(
        overrides={
            "simulation": {"seed": 19},
            "planner": {"belief": {"backend": "ensemble_fit", "predictor": "ensemble"}},
            "output": {"resume": None, "quick_load": False},
        }
    )
    dataset = TrajectoryDataset(tmp_path, config, 12)
    fit = make_learner(tmp_path, horizon=3)
    record = trajectory(mu=0.4, count=20)
    dataset.add(record)
    fit.observe(record)
    fit.update()
    coefficients, probabilities = (
        np.asarray(fit.snapshot.weights),
        np.asarray(fit.snapshot.categorical_weights),
    )
    restored = make_learner(tmp_path, horizon=3)
    restored.restore(tmp_path, 0.02)
    assert restored.generation == 1
    np.testing.assert_array_equal(restored.snapshot.weights, coefficients)
    np.testing.assert_array_equal(restored.snapshot.categorical_weights, probabilities)
    np.testing.assert_allclose(restored.backend.weights, coefficients, atol=1e-7)
    changed = make_learner(tmp_path, horizon=10)
    changed.restore(tmp_path, 0.02)
    assert changed.generation == 2 and changed.last_fit["horizon"] == 10
    categorical = make_learner(tmp_path, backend_name="model_bank")
    categorical.restore(tmp_path, 0.02)
    assert categorical.generation == 3 and categorical.snapshot.weight_kind == "model_probabilities"
    switched_back = make_learner(tmp_path)
    switched_back.restore(tmp_path, 0.02)
    assert switched_back.generation == 4 and switched_back.snapshot.backend == "ensemble_fit"
    state = json.loads((tmp_path / "dynamics_belief.json").read_text())
    assert state["weight_kind"] == "ensemble_coefficients" and state["posterior_available"]


@pytest.mark.parametrize(
    "settings",
    [
        {"fit_horizon": 0},
        {"fit_horizon": True},
        {"fit_max_iterations": 0},
        {"fit_tolerance": 0},
        {"fit_tolerance": float("nan")},
        {"backend": "ensemble_fit", "predictor": "map"},
    ],
)
def test_new_config_options_are_validated(settings):
    with pytest.raises(ValueError, match="planner.belief"):
        load_config(overrides={"planner": {"belief": settings}})
