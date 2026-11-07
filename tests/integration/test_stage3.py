from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.config.schema import RiskConfig, ValueConfig
from sit_lmpc.data.checkpoints import load_risk, load_value, save_value
from sit_lmpc.data.initialization import load_initial, restore_safe_set, save_initial
from sit_lmpc.data.trajectories import AttemptCollector, save_trajectory
from sit_lmpc.experiments.randomness import ExperimentRandomness
from sit_lmpc.planners.lmppi import value_query_inputs
from sit_lmpc.planners.safe_set import SafeSet
from sit_lmpc.training.features import features
from sit_lmpc.training.risk import (
    RiskReplay,
    RiskTrainer,
    balanced_cutoff_samples,
    calibration_metrics,
    fit_risk,
    initial_state,
    predict_risk,
    safe_risk_cutoff,
    terminal_risk,
    trajectory_inputs,
)
from sit_lmpc.training.value import fit_value, predict_value
from sit_lmpc.training.value import initial_state as initial_value


def attempt(number, outcome, steps=3, start=0, action=None):
    state = np.asarray([start, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    frenet = state.copy()
    frenet[0] = 0
    collector = AttemptCollector(number, number, state, frenet, steps)
    for index in range(steps):
        state = state.copy()
        state[0] += 1
        frenet = state.copy()
        frenet[0] = (index + 1) % steps
        control = np.asarray([index + 1.0, -index - 1.0]) if action is None else np.asarray(action)
        collector.append(state, frenet, control, (index + 1) * 0.1, "lmppi", 1, index)
    return collector.finish(outcome, "lap_completed" if outcome == "success" else "out_of_bounds")


def add(replay, rng, trajectory):
    path = (
        replay.directory
        / "trajectories"
        / trajectory.outcome
        / f"attempt-{trajectory.attempt_id:06d}.npz"
    )
    save_trajectory(path, trajectory)
    replay.add(trajectory, path, rng)
    return path


def small_risk(**changes):
    return replace(
        RiskConfig(
            hidden_dims=8,
            layers=1,
            batch_size=20,
            batches_per_epoch=1,
            epochs=3,
            validation_fraction=0,
        ),
        **changes,
    )


@pytest.mark.parametrize("mode", ["state", "state_action"])
def test_risk_pairing_pre_action_phase_and_terminal_semantics(mode):
    trajectory = attempt(1, "failure", steps=4)
    inputs = trajectory_inputs(trajectory, mode)
    np.testing.assert_array_equal(inputs[:, 0], np.arange(4))
    if mode == "state_action":
        np.testing.assert_array_equal(inputs[:, 5:], trajectory.controls)
    np.testing.assert_array_equal(
        terminal_risk(jnp.array([True, False]), jnp.array([False, True])), [1, 0]
    )
    assert np.isnan(terminal_risk(False, False))
    with pytest.raises(ValueError, match="aligned"):
        features(jnp.zeros((3, 7)), "state_action")


@pytest.mark.parametrize("mean", [False, True])
def test_value_planner_queries_next_not_incoming_control(mean):
    states = jnp.arange(2 * 3 * 7.0).reshape(2, 3, 7)
    initial = jnp.ones(7) * -10
    actions = jnp.arange(16.0).reshape(2, 4, 2)
    scale = jnp.asarray([0.4, 3.5])
    query, controls = value_query_inputs(states, initial, actions, scale, mean)
    if mean:
        query = np.asarray(query).reshape(2, 3, 7)
        controls = np.asarray(controls).reshape(2, 3, 2)
        np.testing.assert_array_equal(query[:, 0], np.tile(initial, (2, 1)))
        np.testing.assert_array_equal(query[:, 1:], states[:, :-1])
        np.testing.assert_array_equal(controls, actions[:, :3] * scale)
    else:
        np.testing.assert_array_equal(query, states[:, -1])
        np.testing.assert_allclose(controls, actions[:, -1] * scale)


def test_safe_set_extended_actions_and_warmup_roundtrip(tmp_path):
    config = load_config(
        overrides={
            "value": {"input_mode": "state_action"},
            "planner": {"safe_set_size": 1},
            "output": {"quick_load": False},
        }
    )
    safe = SafeSet(config.planner, 0.1, 8)
    first, second = attempt(1, "success", 8), attempt(2, "success", 8, start=8)
    safe.add_lap(first.as_lap(), first)
    safe.add_lap(second.as_lap(), second)
    data = safe.training_data("state_action")
    assert data.shape[1] == 8
    np.testing.assert_array_equal(data[:7, 5:7], first.controls[1:])
    np.testing.assert_array_equal(data[7, 5:7], second.controls[0])
    # The crossing endpoint is no longer misselected as a next-lap prefix row.
    # Every retained prefix state has its actual outgoing control available.
    assert safe.control_masks[0].all()
    np.testing.assert_array_equal(data[8:, 5:7], second.controls[1:])
    assert np.all(np.diff(safe.frenet[0][:, 0]) > 0)
    path = tmp_path / "init.npz"
    save_initial(path, safe, second.cartesian[-1], np.zeros((config.planner.horizon, 2)), config)
    snapshot = load_initial(path, config)
    restored = SafeSet(config.planner, 0.1, 8)
    restore_safe_set(restored, snapshot)
    np.testing.assert_array_equal(restored.training_data("state_action"), data)
    with np.load(path, allow_pickle=False) as payload:
        assert json.loads(str(payload["metadata"]))["schema"] == 2
        assert all(payload[k].dtype != object for k in payload.files)


def test_old_warmup_is_state_only_not_fictional_actions(tmp_path):
    config = load_config(
        overrides={
            "planner": {"safe_set_size": 1},
            "output": {"quick_load": False},
            "value": {"input_mode": "state"},
        }
    )
    safe = SafeSet(config.planner, 0.1, 8)
    safe.frenet = safe.cartesian = [np.zeros((8, 8))]
    path = tmp_path / "init.npz"
    save_initial(path, safe, np.asarray([0, 0, 0, 10, 0, 0, 0]), np.zeros((10, 2)), config)
    load_initial(path, config)
    with pytest.raises(ValueError, match="aligned controls"):
        load_initial(path, replace(config, value=replace(config.value, input_mode="state_action")))


def test_balanced_batches_recover_ten_percent_not_fifty_percent(tmp_path):
    config = small_risk(epochs=160, learning_rate=0.03)
    replay = RiskReplay(tmp_path, config, capacity=4)
    rng = np.random.default_rng(4)
    for index in range(10):
        add(replay, rng, attempt(index + 1, "failure" if index == 0 else "success"))
    plan = replay.prepare(rng)
    assert plan.metadata["batch_outcome_counts"] == {"success": 10, "failure": 10}
    assert plan.metadata["example_loss_weights"] == pytest.approx({"success": 1.8, "failure": 0.2})
    # Deliberately remove predictive information: every example has one input.
    plan = replace(
        plan, inputs=np.broadcast_to([0.0, 0.0, 0.0, 10.0, 0.0], plan.inputs.shape).copy()
    )
    snapshot, losses = fit_risk(initial_state(config, jax.random.key(0)), plan, config, 1)
    prediction = float(predict_risk(snapshot, jnp.asarray([[0.0, 0.0, 0.0, 10.0, 0.0]]))[0])
    assert prediction == pytest.approx(0.1, abs=0.015)
    assert np.isfinite(losses).all()


def test_state_action_risk_learns_action_dependence(tmp_path):
    config = small_risk(input_mode="state_action", epochs=100, learning_rate=0.03)
    replay = RiskReplay(tmp_path, config, 4)
    rng = np.random.default_rng(4)
    for index in range(8):
        failure = index % 2 == 1
        add(
            replay,
            rng,
            attempt(
                index + 1,
                "failure" if failure else "success",
                action=[0.2 if failure else -0.2, 0.0],
            ),
        )
    snapshot, _ = fit_risk(initial_state(config, jax.random.key(0)), replay.prepare(rng), config, 1)
    states = jnp.asarray([[0.0, 0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 10.0, 0.0]])
    p = predict_risk(snapshot, states, jnp.asarray([[-0.2, 0.0], [0.2, 0.0]]))
    assert p[0] < 0.1 and p[1] > 0.9


def test_episode_cap_and_unequal_lengths_have_inverse_inclusion_correction(tmp_path):
    config = small_risk()
    replay = RiskReplay(tmp_path, config, capacity=1)
    rng = np.random.default_rng(9)
    add(replay, rng, attempt(1, "failure", 2))
    add(replay, rng, attempt(2, "success", 4))
    add(replay, rng, attempt(3, "success", 8))
    plan = replay.prepare(rng)
    assert plan.metadata["episode_inclusion_probabilities"] == {"success": 0.5, "failure": 1.0}
    selected_success = plan.metadata["active_attempts"]["success"][0]
    length = 4 if selected_success == 2 else 8
    assert plan.metadata["example_loss_weights"]["success"] == pytest.approx(
        length / (14 * 0.5 * 0.5)
    )
    assert plan.metadata["example_loss_weights"]["failure"] == pytest.approx(2 / (14 * 0.5))
    assert len(list((tmp_path / "trajectories").rglob("*.npz"))) == 3


def test_risk_split_is_episode_level_and_censored_or_external_data_are_excluded(tmp_path):
    config = small_risk(validation_fraction=0.49)
    replay = RiskReplay(tmp_path, config, 2)
    rng = np.random.default_rng(0)
    for index in range(12):
        add(replay, rng, attempt(index + 1, "failure" if index % 2 else "success", 3))
    before = {e.attempt_id: e.validation for e in replay.episodes}
    plan = replay.prepare(rng)
    assert set(plan.metadata["training_attempts"]).isdisjoint(plan.metadata["validation_attempts"])
    assert plan.validation
    assert before == {e.attempt_id: e.validation for e in replay.episodes}
    incomplete = replace(
        attempt(13, "success"),
        outcome="incomplete",
        goal=np.zeros(4, dtype=bool),
        termination_reason="step_limit",
    )
    path = add(replay, rng, incomplete)
    assert all(e.attempt_id != 13 for e in replay.episodes)
    with pytest.raises(ValueError, match="another experiment"):
        replay.add(attempt(14, "failure"), path.parent.parent.parent / "other.npz", rng)


@pytest.mark.parametrize("risk_mode", ["state", "state_action"])
@pytest.mark.parametrize("value_mode", ["state", "state_action"])
def test_both_learners_all_four_modes_checkpoint_and_predictions(tmp_path, risk_mode, value_mode):
    risk_config = small_risk(input_mode=risk_mode)
    trainer = RiskTrainer(risk_config, 2, 5, tmp_path)
    for index in range(4):
        trajectory = attempt(index + 1, "failure" if index % 2 else "success", 3)
        path = tmp_path / "trajectories" / trajectory.outcome / f"attempt-{index + 1:06d}.npz"
        save_trajectory(path, trajectory)
        trainer.observe(trajectory, path)
    assert trainer.start()
    trainer.wait()
    trainer.save_report()
    restored_risk = load_risk(tmp_path, risk_config)
    states = jnp.asarray(attempt(1, "success", 3).frenet[:-1])
    actions = jnp.asarray(attempt(1, "success", 3).controls)
    p = predict_risk(restored_risk, states, actions if risk_mode == "state_action" else None)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_array_equal(
        predict_risk(
            restored_risk,
            states,
            actions if risk_mode == "state_action" else None,
            jnp.array([True, False, False]),
            jnp.array([False, True, False]),
        )[:2],
        [1, 0],
    )
    value_config = ValueConfig(
        input_mode=value_mode,
        hidden_dims=8,
        layers=2,
        bins=4,
        batch_size=8,
        epochs=2,
        boundary_samples=2,
        samples=2,
    )
    inputs = np.asarray(features(states, value_mode, actions))
    data = np.column_stack((inputs, [-3.0, -2.0, -1.0]))
    snapshot, _ = fit_value(
        initial_value(value_config, jax.random.key(2)), data, value_config, 4, 5, 1
    )
    save_value(tmp_path, snapshot, value_config)
    restored_value = load_value(tmp_path, value_config)
    prediction = predict_value(
        restored_value,
        states,
        jax.random.key(3),
        2,
        actions if value_mode == "state_action" else None,
    )
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(
        predict_value(
            snapshot,
            states,
            jax.random.key(3),
            2,
            actions if value_mode == "state_action" else None,
        ),
        prediction,
    )
    assert json.loads((tmp_path / "value_metadata.json").read_text())["input_mode"] == value_mode
    with pytest.raises(ValueError, match="differs"):
        load_risk(
            tmp_path,
            replace(
                risk_config, input_mode="state" if risk_mode == "state_action" else "state_action"
            ),
        )


def test_single_class_does_not_publish_fake_probability_or_block_value(tmp_path):
    trainer = RiskTrainer(small_risk(), 2, 5, tmp_path)
    trajectory = attempt(1, "success")
    path = tmp_path / "trajectories" / "success" / "attempt-000001.npz"
    save_trajectory(path, trajectory)
    trainer.observe(trajectory, path)
    assert not trainer.start() and trainer.snapshot is None and trainer.generation == 0
    trainer.save_report()
    assert not (tmp_path / "risk_model").exists()


def test_pre_cutoff_checkpoint_evaluation_can_still_be_reported(tmp_path):
    trainer = RiskTrainer(small_risk(), 2, 5, tmp_path)
    trainer.evaluation = {
        "status": "insufficient_held_out_outcomes",
        "states": None,
        "episode_starts": None,
    }
    trainer.save_report()
    assert json.loads((tmp_path / "risk_calibration.json").read_text())["safe_cutoff"] is None


def test_risk_failure_never_publishes_partial_snapshot(tmp_path, monkeypatch):
    import sit_lmpc.data.checkpoints as checkpoints

    trainer = RiskTrainer(small_risk(), 2, 5, tmp_path)
    for index, outcome in enumerate(("success", "failure")):
        trajectory = attempt(index + 1, outcome)
        path = tmp_path / "trajectories" / outcome / f"attempt-{index + 1:06d}.npz"
        save_trajectory(path, trajectory)
        trainer.observe(trajectory, path)

    def fail(*args):
        raise OSError("disk error")

    monkeypatch.setattr(checkpoints, "save_risk", fail)
    trainer.start()
    with pytest.raises(RuntimeError, match="Risk-function training failed"):
        trainer.wait()
    assert trainer.snapshot is None


def test_owned_risk_stream_does_not_change_original_control_randomness():
    a, b = ExperimentRandomness(42), ExperimentRandomness(42)
    fork = a.fork("risk")
    fork.host.normal(size=100)
    fork.next_key()
    np.testing.assert_array_equal(a.host.normal(size=100), b.host.normal(size=100))
    np.testing.assert_array_equal(
        jax.random.key_data(a.next_key()), jax.random.key_data(b.next_key())
    )
    c, d = ExperimentRandomness(42).fork("risk"), ExperimentRandomness(42).fork("risk")
    np.testing.assert_array_equal(c.host.normal(size=20), d.host.normal(size=20))


def test_probability_calibration_metrics_are_not_accuracy_only():
    metrics = calibration_metrics([0.1, 0.9], [0, 1])
    assert metrics["brier"] == pytest.approx(0.01)
    assert metrics["accuracy"] == 1
    assert len(metrics["reliability"]) == 2


def test_safe_risk_cutoff_selects_largest_inclusive_supported_threshold():
    scores = [0.1, 0.1, 0.2, 0.3, 0.4, 0.4]
    failures = [0, 0, 0, 1, 0, 1]
    episodes = [1, 1, 2, 3, 4, 5]
    cutoff = safe_risk_cutoff(scores, failures, episodes, 0.75, 3, 2)
    assert cutoff["status"] == "ready"
    assert cutoff["threshold"] == pytest.approx(0.3)
    assert cutoff["selected_states"] == 4
    assert cutoff["observed_safe_fraction"] == pytest.approx(0.75)
    assert cutoff["selected_success_episodes"] == 2
    assert safe_risk_cutoff(scores, failures, episodes, 0.95, 3, 2)["threshold"] == 0.2
    assert safe_risk_cutoff(scores, failures, episodes, 0.95, 5, 2)["status"] == ("target_not_met")
    assert (
        safe_risk_cutoff(scores[:3], failures[:3], episodes[:3], 0.95, 3, 2)["status"]
        == "insufficient_held_out_outcomes"
    )


def test_cutoff_sampling_balances_classes_without_replacement_and_reproduces():
    scores = np.linspace(0, 1, 8)
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1])
    episodes = np.arange(8)
    first = balanced_cutoff_samples(scores, labels, episodes, 42, 3)
    second = balanced_cutoff_samples(scores, labels, episodes, 42, 3)
    for left, right in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    assert np.bincount(first[1], minlength=2).tolist() == [3, 3]
    assert len(np.unique(first[2])) == 6
    assert set(first[2][first[1] == 1]) == {5, 6, 7}
    assert first[3]["available"] == {"success": 5, "failure": 3}
    assert first[3]["sampled"] == {"success": 3, "failure": 3}
    empty = balanced_cutoff_samples(scores[:5], labels[:5], episodes[:5], 42, 3)
    assert not len(empty[0]) and empty[3]["sampled"] == {"success": 0, "failure": 0}


def test_heldout_report_plot_and_balanced_cutoff_keep_raw_brier(tmp_path):
    config = small_risk(validation_fraction=0.49, cutoff_min_states=3)
    trainer = RiskTrainer(config, 2, 5, tmp_path, plots=True)
    for index in range(24):
        outcome = "failure" if index % 3 == 0 else "success"
        trajectory = attempt(
            index + 1, outcome, action=[0.2 if outcome == "failure" else -0.2, 0.0]
        )
        path = tmp_path / "trajectories" / outcome / f"attempt-{index + 1:06d}.npz"
        save_trajectory(path, trajectory)
        trainer.observe(trajectory, path)
    trainer.start()
    trainer.wait()
    trainer.save_report()
    assert trainer.evaluation["status"] == "held_out_both_outcomes"
    assert trainer.evaluation["states"]["brier"] >= 0
    metadata = json.loads((tmp_path / "risk_metadata.json").read_text())
    assert set(metadata["training_attempts"]).isdisjoint(metadata["validation_attempts"])
    assert metadata["evaluation"]["safe_cutoff"]["target_safe_fraction"] == 0.95
    assert metadata["safe_cutoff"] == metadata["evaluation"]["safe_cutoff"]
    sampling = metadata["safe_cutoff"]["sampling"]
    count = min(sampling["available"].values())
    assert sampling["sampled"] == {"success": count, "failure": count}
    assert metadata["evaluation"]["states"]["examples"] == sum(sampling["available"].values())
    assert metadata["safe_cutoff"]["held_out_states"] == 2 * count
    report = json.loads((tmp_path / "risk_calibration.json").read_text())
    assert report["safe_cutoff"] == metadata["evaluation"]["safe_cutoff"]
    samples = trainer.validation_samples()
    assert samples["generation"] == trainer.snapshot.generation
    assert len(samples["predictions"]) == 2 * count
    assert np.bincount(samples["labels"], minlength=2).tolist() == [count, count]
    assert samples["cutoff"] == metadata["safe_cutoff"]["threshold"]
    assert samples["cutoff_status"] == metadata["safe_cutoff"]["status"]
    assert (
        safe_risk_cutoff(
            samples["predictions"],
            samples["labels"],
            samples["episode_ids"],
            config.target_safe_fraction,
            config.cutoff_min_states,
            config.cutoff_min_episodes,
        )["threshold"]
        == samples["cutoff"]
    )
    assert "predictions" not in report["evaluation"]
    assert (tmp_path / "risk_calibration.png").is_file()


@pytest.mark.parametrize("mode", ["manual", "learned"])
def test_manual_validation_plot_samples_include_available_single_outcome(tmp_path, monkeypatch, mode):
    from sit_lmpc.training import risk as risk_training

    config = small_risk(cutoff_mode=mode, manual_cutoff=0.5)
    trajectory = attempt(1, "success", steps=3)
    path = tmp_path / "attempt.npz"
    save_trajectory(path, trajectory)
    snapshot = SimpleNamespace(
        generation=1, input_mode="state", normalization=SimpleNamespace(low=jnp.zeros(6)),
    )
    record = SimpleNamespace(path=path, attempt_id=1, outcome="success")
    monkeypatch.setattr(risk_training, "predict_risk", lambda snapshot, states, actions: jnp.full(len(states), 0.25))
    report, data = risk_training.evaluate(snapshot, [record], config.batch_size, config, return_samples=True)
    if mode == "manual":
        np.testing.assert_allclose(data["predictions"], [0.25, 0.25, 0.25])
        np.testing.assert_array_equal(data["labels"], [0, 0, 0])
        assert data["cutoff"] == 0.5 and data["cutoff_source"] == "manual"
        assert report["safe_cutoff"]["sampling"]["sampled"] == {"success": 3, "failure": 0}
    else:
        assert not len(data["predictions"]) and data["cutoff"] is None
        assert data["cutoff_status"] == "insufficient_held_out_outcomes"


@pytest.mark.parametrize("mode", ["state", "state_action"])
@pytest.mark.parametrize("sit", [False, True])
def test_runner_trains_value_in_both_modes_and_preserves_sit_failure_stop(
    tmp_path, monkeypatch, mode, sit
):
    from sit_lmpc.experiments import runner
    from test_stage2 import setup_run

    config, captured = setup_run(tmp_path, monkeypatch, failures=(30,), max_steps=42, max_laps=4)
    config = replace(
        config,
        value=replace(config.value, input_mode=mode),
        planner=replace(config.planner, sit_lmpc_mode=sit),
        risk=small_risk(),
    )
    result = runner.run(config)
    assert captured["training"]
    assert all(
        data.shape[1] == (8 if mode == "state_action" else 6) for data in captured["training"]
    )
    assert captured["updates"]
    if sit:
        assert result.status == "out_of_bounds" and result.trials == 3
        assert len(captured["resets"]) == 1
    else:
        assert result.status == "lap_limit" and result.trials == 4
        assert len(captured["resets"]) == 2
    summary = json.loads((result.directory / "summary.json").read_text())
    assert summary["risk_training_generations"] >= 1
