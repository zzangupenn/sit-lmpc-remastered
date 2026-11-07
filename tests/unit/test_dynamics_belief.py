from dataclasses import replace, fields
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.adapters.f1tenth import vehicle_parameters
from sit_lmpc.config.schema import BeliefConfig, SimulationConfig
from sit_lmpc.config import load_config
from sit_lmpc.data.dynamics import DynamicsReplay, trajectory_transitions
from sit_lmpc.data.trajectories import AttemptCollector, TrajectoryDataset
from sit_lmpc.dynamics.bank import STModelBank
from sit_lmpc.dynamics.belief import LikelihoodBackend, replay_probabilities, transition_statistics
from sit_lmpc.dynamics.st import STModel, step
from sit_lmpc.experiments.randomness import ExperimentRandomness
from sit_lmpc.training.dynamics import DynamicsBeliefLearner


def model(mu=0.5, dtype="float32", timestep=0.1, substep=0.02):
    sim = SimulationConfig(
        friction=mu,
        dtype=dtype,
        timestep=timestep,
        integrator_timestep=substep,
        action_scale=(0.4, 3.5),
    )
    return STModel.from_vehicle(vehicle_parameters(mu), sim)


def trajectory(
    attempt=1,
    outcome="success",
    count=12,
    mu=0.5,
    steering=0.12,
    control=(0.08, 0.35),
    substep=0.02,
):
    dynamics = model(mu, substep=substep)
    state = np.asarray([0, 0, steering, 12, 0.2, 0, 0], dtype=np.float32)
    frenet = state.copy()
    frenet[0] = 0
    collector = AttemptCollector(
        attempt, attempt, state, frenet, count, integrator_timestep=substep, action_scale=(0.4, 3.5)
    )
    for i in range(count):
        applied = np.asarray(control) + np.asarray([0.002 * (i % 3), 0.005 * (i % 2)])
        state = np.asarray(step(dynamics, state, applied / dynamics.action_scale))
        frenet = state.copy()
        frenet[0] = (i + 1) % count
        if outcome == "failure" and i == count - 1:
            frenet[1] = 4.2
        collector.append(state, frenet, applied, (i + 1) * 0.1, "mppi", 0, i, belief_generation=2)
    return collector.finish(
        outcome,
        "out_of_bounds"
        if outcome == "failure"
        else "lap_completed"
        if outcome == "success"
        else "step_limit",
    )


def learner(tmp_path, *, frictions=(0.4, 0.5, 0.7), **settings):
    config = replace(
        BeliefConfig(), **{"enabled": True, "backend": "likelihood", "batch_size": 8,
                          "predictor": "map", **settings}
    )
    return DynamicsBeliefLearner(
        config, model(0.7), frictions, tmp_path, ExperimentRandomness(19).fork("dynamics_belief")
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_likelihood_matches_physical_control_and_direct_calculation(dtype):
    record = trajectory(count=12, outcome="failure")
    batch = trajectory_transitions(record, 0.01)
    bank = STModelBank.from_model(model(0.7, dtype=dtype), (0.4, 0.5, 0.7))
    config = replace(BeliefConfig(), batch_size=8)
    backend = LikelihoodBackend(bank, config)
    stats = backend.statistics(batch)
    scales = np.asarray(config.residual_scales)
    expected = []
    for params in bank.params:
        member = bank.member(params)
        predictions = np.asarray(
            [
                step(member, state, control / bank.action_scale)
                for state, control in zip(batch.states, batch.controls, strict=True)
            ]
        )
        residual = batch.next_states - predictions
        for component in (2, 4, 6):
            residual[:, component] = np.arctan2(
                np.sin(residual[:, component]), np.cos(residual[:, component])
            )
        expected.append(
            np.mean(-0.5 * np.sum((residual / scales) ** 2 + np.log(2 * np.pi * scales**2), axis=1))
        )
    np.testing.assert_allclose(stats["mean_log_likelihood"], expected, atol=2e-4)
    assert stats["standardized_rmse"][1] < 1e-3
    np.testing.assert_array_equal(batch.controls, record.controls)
    assert len(batch) == 12  # Boundary crossing is useful; no reset transition.


def test_success_and_failure_learn_without_recounting_replay(tmp_path):
    fit = learner(tmp_path, evidence_strength=100)
    pre_update = fit.observe(trajectory())
    assert pre_update["status"] == "pre_update" and fit.generation == 0
    assert fit.update(pre_update) and fit.generation == 1
    first = np.asarray(fit.snapshot.weights).copy()
    assert first[1] > 0.99
    assert fit.update() and fit.generation == 2
    np.testing.assert_array_equal(first, fit.snapshot.weights)
    fit.observe(trajectory(2, "failure"))
    assert fit.update() and fit.generation == 3
    assert fit.predictor(model(0.7)).params[0] == pytest.approx(0.5)


def test_replay_recent_history_are_disjoint_and_seeded():
    replay = DynamicsReplay()
    for i, outcome in enumerate(("success", "failure", "success", "failure", "incomplete"), 1):
        record = trajectory(i, outcome, count=12)
        replay.add(i, outcome, trajectory_transitions(record, 0.02))
    selected = replay.select(np.random.default_rng(7), 5, 0.5)
    again = replay.select(np.random.default_rng(7), 5, 0.5)
    np.testing.assert_array_equal(selected.transitions.ids, again.transitions.ids)
    assert selected.recent_count == 24 and selected.history_count == 5
    assert set(selected.transitions.ids[:24, 0]) == {3, 4}
    assert not set(selected.transitions.ids[24:, 0]) & {3, 4}
    assert len(np.unique(selected.transitions.ids, axis=0)) == 29
    assert selected.weights[:24].sum() == pytest.approx(0.5)
    assert selected.weights[24:].sum() == pytest.approx(0.5)


def test_empty_missing_cohorts_and_incomplete_contact_filter():
    replay = DynamicsReplay()
    assert len(replay.select(np.random.default_rng(1), 10, 0.5).transitions) == 0
    record = trajectory(outcome="incomplete")
    states = record.cartesian.copy()
    states[-1, 0] = np.nan
    contacts = record.contacts.copy()
    contacts[2] = True
    batch = trajectory_transitions(replace(record, cartesian=states, contacts=contacts), 0.02)
    assert len(batch) == 10 and 2 not in batch.ids[:, 1] and 11 not in batch.ids[:, 1]
    replay.add(1, "incomplete", batch)
    selected = replay.select(np.random.default_rng(1), 20, 1.0)
    assert selected.recent_count == 0 and selected.weights.sum() == pytest.approx(1)


def test_log_normalization_preserves_zero_prior_and_weak_information(tmp_path):
    p = replay_probabilities([0.1, 0.9, 0], [-1e6, -1e6, 1e6], 32)
    np.testing.assert_allclose(p, [0.1, 0.9, 0], atol=1e-9)
    np.testing.assert_allclose(replay_probabilities([0.5, 0.5], [1e6, 1e6], 1e300), [0.5, 0.5])
    fit = learner(tmp_path, prior=(0.2, 0.3, 0.5))
    record = trajectory(steering=0, control=(0, 0), count=1, outcome="incomplete")
    fit.observe(record)
    fit.update()
    np.testing.assert_allclose(fit.snapshot.weights, [0.2, 0.3, 0.5], atol=1e-6)


def test_all_models_mismatch_keeps_last_valid_snapshot(tmp_path):
    fit = learner(tmp_path)
    fit.observe(trajectory())
    fit.update()
    old = fit.snapshot
    record = trajectory(2, count=12)
    fit.observe(replace(record, cartesian=record.cartesian + np.arange(13)[:, None] * 100))
    assert not fit.update()
    assert fit.snapshot is old and fit.last_fit["status"] == "all_models_mismatch"
    assert (
        json.loads((tmp_path / "dynamics_belief.json").read_text())["last_fit"]["status"]
        == "all_models_mismatch"
    )


def test_wrong_prior_support_is_reported_without_inventing_new_models(tmp_path):
    fit = learner(tmp_path, frictions=(0.1, 0.2), residual_scales=(0.00001,) * 7)
    fit.observe(trajectory(mu=0.55))
    assert not fit.update()
    assert fit.generation == 0 and fit.last_fit["status"] == "all_models_mismatch"
    np.testing.assert_allclose(fit.snapshot.weights, [0.5, 0.5])


def test_option_five_tracks_published_weights_and_predictor(tmp_path):
    from sit_lmpc.diagnostics.model_comparison import ModelComparisons

    fit = learner(tmp_path)
    record = trajectory()
    fit.observe(record)
    fit.update()
    comparison = ModelComparisons(model(0.7), model(0.55), (0.4, 0.5, 0.7), 3)
    comparison.set_belief(fit.predictor(model(0.7)), fit.snapshot)
    data = comparison.planned(record.cartesian[0], np.zeros((3, 2)), 0)
    np.testing.assert_array_equal(data.model_weights, fit.snapshot.weights)
    assert data.belief_generation == 1 and data.belief_friction == pytest.approx(0.5)
    assert not list(tmp_path.glob("model_comparison*"))


def test_off_grid_truth_is_finite_bank_approximation(tmp_path):
    fit = learner(tmp_path, evidence_strength=100)
    fit.observe(trajectory(mu=0.55, count=20))
    assert fit.update()
    weights = np.asarray(fit.snapshot.weights)
    assert np.isfinite(weights).all() and weights.sum() == pytest.approx(1)
    assert weights[1] > weights[2] and weights[1] > weights[0]
    assert fit.snapshot.bank.params[:, 0].tolist() == pytest.approx([0.4, 0.5, 0.7])


def test_one_model_and_fixed_predictor_modes(tmp_path):
    fit = learner(tmp_path, frictions=(0.5,), predictor="fixed")
    fit.observe(trajectory())
    fit.update()
    np.testing.assert_array_equal(fit.snapshot.weights, [1])
    fixed = model(0.7)
    assert fit.predictor(fixed) is fixed


def test_angular_wrap_and_fixed_padded_batches_reuse_compilation():
    transition_statistics.clear_cache()
    bank = STModelBank.from_model(model(), (0.4, 0.5, 0.7))
    backend = LikelihoodBackend(bank, replace(BeliefConfig(), batch_size=8))
    batch = trajectory_transitions(trajectory(count=12), 0.02)
    original = backend.statistics(batch)
    targets = batch.next_states.copy()
    targets[:, 4] += 2 * np.pi
    wrapped = backend.statistics(replace(batch, next_states=targets).take(np.arange(7)))
    assert wrapped["standardized_rmse"][1] < 0.001
    assert original["standardized_rmse"][1] < 0.001
    assert transition_statistics._cache_size() == 1


def test_resume_restores_and_refits_changed_settings_with_old_integration_context(tmp_path):
    config = load_config(
        overrides={
            "simulation": {"seed": 7, "integrator_timestep": 0.01},
            "planner": {"belief": {"enabled": True}},
            "output": {"quick_load": False, "resume": None},
        }
    )
    dataset = TrajectoryDataset(tmp_path, config, 12)
    fit = learner(tmp_path)
    for i, outcome in enumerate(("success", "failure"), 1):
        record = trajectory(i, outcome, substep=0.01)
        dataset.add(record)
        fit.observe(record)
        fit.update()
    restored = learner(tmp_path)
    restored.restore(tmp_path, 0.02)
    np.testing.assert_array_equal(restored.snapshot.weights, fit.snapshot.weights)
    assert restored.generation == fit.generation == 2
    np.testing.assert_array_equal(restored.replay.last_selection, fit.replay.last_selection)
    assert restored.replay.episodes[1][1].substeps.tolist() == [0.01] * 12
    changed = learner(tmp_path, frictions=(0.45, 0.5, 0.65, 0.7), evidence_strength=64)
    changed.restore(tmp_path, 0.02)
    assert changed.generation == 3 and len(changed.snapshot.weights) == 4


@pytest.mark.parametrize(
    "settings",
    [
        {"predictor": "mean"},
        {"backend": "unknown"},
        {"prior": [0, 0, 0, 0]},
        {"prior": [1]},
        {"history_samples": 0},
        {"recent_fraction": 1.1},
        {"residual_scales": [0.1]},
        {"evidence_strength": 0},
    ],
)
def test_belief_config_validation(settings):
    with pytest.raises(ValueError, match="planner.belief"):
        load_config(overrides={"planner": {"model_bank": {"frictions": [0.3, 0.4, 0.7, 0.8]}, "belief": settings}})


def test_default_preset_selects_ensemble_and_belief_config_is_hashable():
    config = load_config()
    assert config.planner.belief.predictor == "ensemble"
    hash(config.planner)
    restored = load_config(
        overrides={
            "planner": {
                "belief": {
                    "residual_scales": [0.1] * 7,
                    "prior": [1] * len(config.planner.model_bank.frictions),
                }
            }
        }
    )
    hash(restored.planner)
    assert set(config.as_dict()["planner"]["belief"]) == {
        field.name for field in fields(BeliefConfig)
    }


@pytest.mark.parametrize("backend", ["model_bank", "ensemble_fit"])
def test_quick_load_known_warmup_pairs_are_deduplicated_and_checkpointed(tmp_path, backend):
    from sit_lmpc.data.initialization import InitialSnapshot

    record = trajectory(count=12)
    states = np.column_stack((record.cartesian[1:], record.times[1:]))
    controls = np.vstack((record.controls[1:], np.zeros((1, 2))))
    mask = np.arange(12) < 11
    initial = InitialSnapshot(
        (states, states),
        (states, states),
        record.cartesian[-1],
        np.zeros((10, 2)),
        {"config": {"simulation": {"timestep": 0.1, "integrator_timestep": 0.02}}},
        (controls, controls),
        (mask, mask),
    )
    fit = learner(tmp_path, backend=backend, predictor="ensemble")
    fit.import_initial(initial)
    assert len(fit.replay.episodes[-1][1]) == 11 and fit.generation == 1
    restored = learner(tmp_path, backend=backend, predictor="ensemble")
    restored.restore(tmp_path, 0.02)
    np.testing.assert_array_equal(restored.snapshot.weights, fit.snapshot.weights)
    np.testing.assert_array_equal(restored.replay.episodes[-1][1].controls, controls[:-1])
