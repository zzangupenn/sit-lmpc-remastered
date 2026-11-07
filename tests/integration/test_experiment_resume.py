"""Trial-boundary continuation with explicit, small test configurations."""

from dataclasses import replace
import json
import shutil
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.data.trajectories import load_trajectories
from sit_lmpc.data.checkpoints import load_risk, load_value
from sit_lmpc.experiments import runner
from sit_lmpc.training.value import ValueTrainer
from sit_lmpc.training.risk import RiskTrainer
from test_stage2 import setup_run


def test_resume_creates_new_run_and_keeps_attempt_ids(tmp_path, monkeypatch):
    config, _ = setup_run(
        tmp_path, monkeypatch, max_steps=14, max_laps=3, quick_load=False, failures=()
    )
    config = replace(
        config,
        risk=replace(config.risk, enabled=False, input_mode="state"),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config, tracking_only=True)
    assert first.status == "step_limit" and first.trials == 1
    assert (first.directory / "checkpoints" / "trial-000001" / "manifest.json").is_file()
    assert [t.attempt_id for t in load_trajectories(first.directory, "incomplete")] == [2]

    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=25),
        output=replace(config.output, resume=str(first.directory / "checkpoints" / "trial-000001")),
    )
    second = runner.run(resumed, tracking_only=True)
    assert second.status == "step_limit" and second.trials == 2
    assert second.directory != first.directory
    assert second.directory.parent == tmp_path / "custom1"
    assert second.directory.name.startswith("test-resume-20")
    assert [t.attempt_id for t in load_trajectories(second.directory, "success")] == [1, 2]
    assert [t.attempt_id for t in load_trajectories(second.directory, "incomplete")] == [3]
    assert [t.attempt_id for t in load_trajectories(first.directory, "incomplete")] == [2]
    assert not (second.directory / "recovery").exists()
    assert not (first.directory / "recovery").exists()
    source_metadata = json.loads((first.directory / "metadata.json").read_text())
    new_metadata = json.loads((second.directory / "metadata.json").read_text())
    new_context = json.loads((second.directory / "trajectories" / "context.json").read_text())
    assert "reference_files" not in source_metadata
    assert "reference_files" not in new_metadata
    assert new_metadata["checkpoint_trial"] == 1
    assert new_metadata["resumed_from_checkpoint"] == str(
        first.directory / "checkpoints" / "trial-000001"
    )
    assert new_context["config"]["output"]["directory"] == str(second.directory)
    assert new_context["config"]["output"]["resume"] is None
    summary = json.loads((second.directory / "summary.json").read_text())
    assert summary["trials"] == 2 and summary["attempts"] == 3


def test_invalid_resume_configuration_does_not_create_a_new_run(tmp_path, monkeypatch):
    config, _ = setup_run(tmp_path, monkeypatch, max_steps=2, max_laps=2)
    config = replace(config, risk=replace(config.risk, enabled=False, input_mode="state"))
    first = runner.run(config, tracking_only=True)
    incompatible = replace(
        config,
        simulation=replace(config.simulation, map="Spielberg"),
        output=replace(config.output, resume=str(first.directory / "checkpoints" / "trial-000000")),
    )
    try:
        runner.run(incompatible, tracking_only=True)
    except ValueError as error:
        assert "saved experiment configuration" in str(error)
    else:
        raise AssertionError("changing the map would invalidate restored Frenet trajectories")
    assert not (first.directory / "recovery").exists()
    assert not (tmp_path / "custom1").exists()


def test_resumed_pending_lap_keeps_continuous_finish_progress(tmp_path, monkeypatch):
    config, _ = setup_run(
        tmp_path, monkeypatch, max_steps=25, max_laps=2, quick_load=False, failures=()
    )
    first = runner.run(config, tracking_only=True)
    assert first.status == "lap_limit" and first.trials == 2
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=37, max_laps=3),
        output=replace(config.output, resume=str(first.directory / "checkpoints" / "trial-000002")),
    )
    second = runner.run(resumed, tracking_only=True)
    assert second.status == "lap_limit" and second.trials == 3
    with np.load(second.directory / "checkpoints" / "trial-000003" / "state.npz") as data:
        record = data["safe_0_frenet"]
    np.testing.assert_allclose(record[:12, 0], np.arange(1, 13))
    assert np.all(np.diff(record[:, 0]) > 0)
    assert record[:, -1].max() < 0.4  # Only the contiguous next-lap prefix.
    trajectories = load_trajectories(second.directory, "success")
    assert len(trajectories) == 3
    assert all(len(t.controls) == 12 and t.frenet[-1, 0] == 0 for t in trajectories)


def test_resume_after_failed_trial_restarts_at_failed_lap_start(tmp_path, monkeypatch):
    config, _ = setup_run(
        tmp_path, monkeypatch, failures=(3,), max_steps=4, max_laps=2, quick_load=False
    )
    config = replace(
        config,
        risk=replace(config.risk, enabled=False, input_mode="state"),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config, tracking_only=True)
    assert first.trials == first.failures == 1
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=15),
        output=replace(config.output, resume=str(first.directory / "checkpoints" / "trial-000001")),
    )
    second = runner.run(resumed, tracking_only=True)
    assert second.status == "lap_limit" and second.trials == 2
    assert second.directory != first.directory
    assert [t.attempt_id for t in load_trajectories(first.directory, "failure")] == [1]
    failures = load_trajectories(second.directory, "failure")
    successes = load_trajectories(second.directory, "success")
    assert [t.attempt_id for t in failures] == [1]
    assert [t.attempt_id for t in successes] == [2]
    assert successes[0].cartesian[0, 0] == failures[0].cartesian[0, 0]


def test_corrupt_selected_checkpoint_requires_explicit_previous_checkpoint(tmp_path, monkeypatch):
    config, _ = setup_run(
        tmp_path, monkeypatch, max_steps=14, max_laps=3, quick_load=False, failures=()
    )
    config = replace(
        config,
        risk=replace(config.risk, enabled=False, input_mode="state"),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config, tracking_only=True)
    latest = first.directory / "checkpoints" / "trial-000001" / "state.json"
    latest.write_text("damaged")
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=15),
        output=replace(config.output, resume=str(first.directory / "checkpoints" / "trial-000001")),
    )
    with pytest.raises(ValueError, match="checksum"):
        runner.run(resumed, tracking_only=True)
    assert not (first.directory / "recovery").exists()
    resumed = replace(
        resumed,
        output=replace(
            resumed.output, resume=str(first.directory / "checkpoints" / "trial-000000")
        ),
    )
    second = runner.run(resumed, tracking_only=True)
    assert second.trials == 1
    assert second.directory != first.directory
    assert latest.read_text() == "damaged"
    assert (first.directory / "trajectories" / "success" / "attempt-000001.npz").is_file()
    assert not (second.directory / "recovery").exists()


@pytest.mark.parametrize("mode", ["state", "state_action"])
def test_resume_restores_trained_value_and_continues_generation(tmp_path, monkeypatch, mode):
    config, captured = setup_run(
        tmp_path, monkeypatch, max_steps=25, max_laps=4, quick_load=False, failures=()
    )
    monkeypatch.setattr(runner, "ValueTrainer", ValueTrainer)
    config = replace(
        config,
        risk=replace(config.risk, enabled=False, input_mode="state"),
        value=replace(
            config.value,
            input_mode=mode,
            hidden_dims=8,
            layers=2,
            bins=4,
            batch_size=16,
            epochs=1,
            boundary_samples=4,
            samples=2,
        ),
        output=replace(config.output, checkpoint_every_trials=2),
    )
    first = runner.run(config)
    assert first.trials == 2 and first.training_generations == 1
    checkpoint = first.directory / "checkpoints" / "trial-000002"
    assert int(load_value(checkpoint, config.value).generation) == 1
    portable = tmp_path / "copied-checkpoint"
    shutil.copytree(checkpoint, portable)
    first.directory.rename(tmp_path / "source-unavailable")
    monkeypatch.setattr(runner, "map_path", lambda *args: pytest.fail("resume used external map"))

    resumed = replace(
        config,
        simulation=replace(
            config.simulation, max_steps=37, max_laps=3, friction=0.8,
            initial_speed=12.0, action_scale=(0.8, 7.0), seed=19,
        ),
        planner=replace(config.planner, friction=0.7, horizon=4, samples=16, lambda_count=2),
        value=replace(config.value, learning_rate=0.002, batch_size=8, epochs=2, samples=3),
        output=replace(config.output, resume=str(portable), exp_name="branch-lr"),
    )
    second = runner.run(resumed)
    assert second.trials == 3 and second.laps == 3
    assert second.directory != first.directory
    assert second.training_generations == 2 and second.learned_steps > first.learned_steps
    assert captured["learned"].actions.shape == (4 + int(mode == "state_action"), 2)
    model_metadata = json.loads((second.directory / "value_metadata.json").read_text())
    assert model_metadata["value_config"]["learning_rate"] == 0.002
    assert second.directory.name.startswith("branch-lr-")
    assert [t.attempt_id for t in load_trajectories(second.directory, "success")] == [1, 2, 3]


@pytest.mark.parametrize("mode", ["state", "state_action"])
def test_resume_restores_risk_replay_and_generation(tmp_path, monkeypatch, mode):
    config, _ = setup_run(
        tmp_path, monkeypatch, failures=(3,), max_steps=16, max_laps=4, quick_load=False
    )
    config = replace(
        config,
        risk=replace(
            config.risk,
            enabled=True,
            input_mode=mode,
            hidden_dims=8,
            layers=1,
            batch_size=8,
            batches_per_epoch=1,
            epochs=1,
            validation_fraction=0.0,
        ),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config)
    assert first.trials == 2 and first.failures == 1
    checkpoint = first.directory / "checkpoints" / "trial-000002"
    assert int(load_risk(checkpoint, config.risk).generation) == 1
    state = json.loads((checkpoint / "state.json").read_text())
    cutoff = json.loads((checkpoint / "risk_metadata.json").read_text())["safe_cutoff"]
    assert cutoff["generation"] == state["risk_generation"]
    assert cutoff == state["risk_evaluation"]["safe_cutoff"]
    assert [record["attempt_id"] for record in state["risk_replay"]] == [1, 2]
    portable = tmp_path / "copied-checkpoint"
    shutil.copytree(checkpoint, portable)
    first.directory.rename(tmp_path / "source-unavailable")
    monkeypatch.setattr(runner, "map_path", lambda *args: pytest.fail("resume used external map"))

    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=28),
        risk=replace(
            config.risk, learning_rate=0.002, batch_size=16, epochs=2,
            buffer_size=2, failure_fraction=0.25, validation_fraction=0.4,
        ),
        output=replace(config.output, resume=str(portable)),
    )
    second = runner.run(resumed)
    assert second.trials == 3 and second.failures == 1
    resumed_checkpoint = second.directory / "checkpoints" / "trial-000003"
    assert (resumed_checkpoint / "manifest.json").is_file()
    assert (resumed_checkpoint / "map" / "custom1_map.yaml").is_file()
    summary = json.loads((second.directory / "summary.json").read_text())
    assert summary["risk_training_generations"] == 2
    assert [t.attempt_id for t in load_trajectories(second.directory, "success")] == [2, 3]
    metadata = json.loads((second.directory / "risk_metadata.json").read_text())
    assert metadata["risk_config"]["learning_rate"] == 0.002
    replay = json.loads((resumed_checkpoint / "state.json").read_text())["risk_replay"]
    assert [entry["validation"] for entry in replay[:2]] == [False, False]


@pytest.mark.parametrize("old_mode,new_mode", [("learned", "manual"), ("manual", "manual"), ("manual", "learned")])
def test_changed_resume_cutoff_updates_restored_model_and_plot_before_training(tmp_path, monkeypatch, old_mode, new_mode):
    config, _ = setup_run(
        tmp_path, monkeypatch, failures=(3,), max_steps=16, max_laps=4, quick_load=False
    )
    config = replace(
        config,
        planner=replace(config.planner, risk_constraint=replace(config.planner.risk_constraint, enabled=True)),
        risk=replace(
            config.risk, enabled=True, input_mode="state", hidden_dims=8, layers=1,
            batch_size=8, batches_per_epoch=1, epochs=1, validation_fraction=0,
            cutoff_mode=old_mode, manual_cutoff=0.8, target_safe_fraction=0.95,
        ),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config)
    checkpoint = first.directory / "checkpoints" / "trial-000002"
    original_metadata = (checkpoint / "risk_metadata.json").read_bytes()
    original_risk = load_risk(checkpoint, config.risk)
    assert original_risk.generation == 1
    trainers = []

    class RecordingTrainer(RiskTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)

    monkeypatch.setattr(runner, "RiskTrainer", RecordingTrainer)
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=17),
        risk=replace(config.risk, cutoff_mode=new_mode, manual_cutoff=0.5),
        output=replace(config.output, resume=str(checkpoint)),
    )
    second = runner.run(resumed)
    trainer = trainers[-1]
    assert trainer.generation == 1
    assert int(trainer.snapshot.state.step) == int(original_risk.state.step)
    samples = trainer.validation_samples()
    metadata = json.loads((second.directory / "risk_metadata.json").read_text())
    assert metadata["risk_config"]["cutoff_mode"] == new_mode
    assert metadata["risk_config"]["manual_cutoff"] == 0.5
    assert metadata["safe_cutoff"] == trainer.evaluation["safe_cutoff"]
    if new_mode == "manual":
        assert trainer.constraint_cutoff["threshold"] == samples["cutoff"] == 0.5
        assert trainer.constraint_snapshot.generation == samples["generation"] == 1
        active = json.loads((second.directory / "risk_constraint" / "risk_metadata.json").read_text())
        assert active["safe_cutoff"] == trainer.constraint_cutoff
    else:
        assert trainer.constraint_snapshot is None and trainer.constraint_cutoff is None
        assert samples["cutoff"] is None
        assert not (second.directory / "risk_constraint").exists()
    events = [json.loads(line) for line in (second.directory / "events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "risk_cutoff_updated" and event["mode"] == new_mode for event in events)
    assert (checkpoint / "risk_metadata.json").read_bytes() == original_metadata


@pytest.mark.parametrize("count", [1, 4])
def test_restored_action_horizon_preserves_physical_prefix_and_pads_new_steps(count):
    saved = jnp.asarray([[0.2, -0.4], [0.6, 0.8]])
    restored = np.asarray(runner.restore_actions(saved, jnp.zeros((count, 2)), (0.4, 3.5), (0.8, 7.0)))
    overlap = min(count, len(saved))
    np.testing.assert_allclose(restored[:overlap] * [0.8, 7.0], np.asarray(saved[:overlap]) * [0.4, 3.5])
    np.testing.assert_array_equal(restored[overlap:], np.zeros((count - overlap, 2)))


def test_resume_can_change_precision_without_resetting_risk_optimizer(tmp_path, monkeypatch):
    config, _ = setup_run(tmp_path, monkeypatch, failures=(3,), max_steps=16, max_laps=4)
    config = replace(
        config,
        simulation=replace(config.simulation, dtype="float32"),
        risk=replace(config.risk, enabled=True, input_mode="state", hidden_dims=8, layers=1,
                     batch_size=8, batches_per_epoch=1, epochs=1, validation_fraction=0),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config)
    checkpoint = first.directory / "checkpoints" / "trial-000002"
    original = load_risk(checkpoint, config.risk)
    trainers = []

    class RecordingTrainer(RiskTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)

    monkeypatch.setattr(runner, "RiskTrainer", RecordingTrainer)
    resumed = replace(config, simulation=replace(config.simulation, dtype="float64", max_steps=17),
                      output=replace(config.output, resume=str(checkpoint)))
    with jax.enable_x64():
        second = runner.run(resumed)
        snapshot = trainers[-1].snapshot
        assert snapshot.generation == original.generation
        assert int(snapshot.state.step) == int(original.state.step)
        assert snapshot.normalization.low.dtype == jnp.float64
        floats = [leaf for leaf in jax.tree.leaves(snapshot) if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)]
        assert all(leaf.dtype == jnp.float64 for leaf in floats)
        assert second.trials == first.trials


def test_disabling_risk_on_resume_keeps_model_and_replay_for_later_enable(tmp_path, monkeypatch):
    config, _ = setup_run(tmp_path, monkeypatch, failures=(3,), max_steps=16, max_laps=4)
    monkeypatch.setattr(runner, "ValueTrainer", ValueTrainer)
    config = replace(
        config,
        risk=replace(config.risk, enabled=True, input_mode="state", hidden_dims=8, layers=1,
                     batch_size=8, batches_per_epoch=1, epochs=1, validation_fraction=0),
        value=replace(config.value, hidden_dims=8, layers=2, bins=4, batch_size=8,
                      epochs=1, boundary_samples=0, samples=2),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    first = runner.run(config)
    disabled = replace(
        config, simulation=replace(config.simulation, max_steps=28),
        risk=replace(config.risk, enabled=False),
        output=replace(config.output, directory=None, resume=str(first.directory / "checkpoints" / "trial-000002"),
                       exp_name="risk-off"),
    )
    second = runner.run(disabled)
    checkpoint = second.directory / "checkpoints" / "trial-000003"
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["risk_generation"] == 1 and len(state["risk_replay"]) == 3
    assert (checkpoint / "risk_model").is_dir()
    enabled = replace(
        disabled, simulation=replace(disabled.simulation, max_steps=40),
        risk=replace(disabled.risk, enabled=True),
        output=replace(disabled.output, directory=None, resume=str(checkpoint), exp_name="risk-on"),
    )
    third = runner.run(enabled)
    summary = json.loads((third.directory / "summary.json").read_text())
    assert third.trials == 4 and summary["risk_training_generations"] == 2


def test_quick_load_init_stays_separate_from_checkpoint_resume(tmp_path, monkeypatch):
    config, captured = setup_run(
        tmp_path, monkeypatch, quick_load=True, failures=(), max_steps=2, max_laps=2
    )
    monkeypatch.setattr(runner, "ValueTrainer", ValueTrainer)
    config = replace(
        config,
        risk=replace(config.risk, enabled=False, input_mode="state"),
        value=replace(
            config.value,
            input_mode="state",
            hidden_dims=8,
            layers=2,
            bins=4,
            batch_size=16,
            epochs=1,
            boundary_samples=4,
            samples=2,
        ),
        output=replace(config.output, checkpoint_every_trials=1),
    )
    original_init = (tmp_path / "warmup.npz").read_bytes()
    first = runner.run(config)
    assert first.training_generations == 1
    checkpoint = first.directory / "checkpoints" / "trial-000000"
    assert int(load_value(checkpoint, config.value).generation) == 1
    portable = tmp_path / "copied-checkpoint"
    shutil.copytree(checkpoint, portable)
    first.directory.rename(tmp_path / "source-unavailable")
    (tmp_path / "warmup.npz").rename(tmp_path / "warmup-retained.npz")
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=4),
        output=replace(config.output, resume=str(portable)),
    )
    second = runner.run(resumed)
    assert second.training_generations == 1 and second.learned_steps == 4
    assert (tmp_path / "warmup-retained.npz").read_bytes() == original_init
    assert captured["precompiled"]
