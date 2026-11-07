from dataclasses import replace
import json
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.data.trajectories import (
    AttemptCollector,
    TrajectoryDataset,
    load_trajectory,
    load_trajectories,
    save_trajectory,
)


def attempt(outcome="failure", start_s=0):
    initial = np.asarray([start_s, 0, 0.1, 10, 0.2, 0.03, 0.04])
    length = 3 if outcome == "success" else 12
    collector = AttemptCollector(1, 1, initial, initial, length)
    for i in range(1, 4):
        state = initial.copy()
        state[0] = (start_s + i) % length
        state[1] = -4.2 if outcome == "failure" and i == 3 else 0
        collector.append(state, state, np.asarray([0.1, i]), i / 10, "lmppi", 2, i - 1)
    return collector


@pytest.mark.parametrize("outcome", ["success", "failure", "incomplete"])
def test_full_trajectory_roundtrip_and_separate_terminal_flags(tmp_path, outcome):
    record = attempt(outcome).finish(outcome, outcome)
    path = tmp_path / "record.npz"
    save_trajectory(path, record)
    restored = load_trajectory(path)
    assert restored.cartesian.shape == restored.frenet.shape == (4, 7)
    assert restored.controls.shape == (3, 2)
    np.testing.assert_array_equal(restored.cartesian[0], [0, 0, 0.1, 10, 0.2, 0.03, 0.04])
    np.testing.assert_array_equal(restored.times, [0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(restored.progress, [0, 1, 2, 3])
    np.testing.assert_array_equal(
        restored.physical_failure, [False, False, False, outcome == "failure"]
    )
    np.testing.assert_array_equal(restored.goal, [False, False, False, outcome == "success"])
    np.testing.assert_array_equal(restored.value_generations, [2, 2, 2])
    assert restored.outcome == outcome and restored.controllers.tolist() == ["lmppi"] * 3
    np.testing.assert_array_equal(restored.as_lap().cartesian, restored.cartesian[1:])
    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
    with pytest.raises(FileExistsError):
        save_trajectory(path, record)


def test_progress_requires_full_lap_not_initial_or_early_finish_line_crossing():
    collector = attempt("incomplete", start_s=0)
    assert collector.progress == [0, 1, 2, 3] and not collector.completed_lap
    for i in range(4, 13):
        state = collector.frenet[-1].copy()
        state[0] = i % 12
        collector.append(state, state, np.zeros(2), i / 10, "mppi", 0, i - 1)
    assert collector.completed_lap and collector.progress[-1] == 12


def test_sampled_finish_line_overshoot_does_not_drift_lap_boundaries():
    initial = np.asarray([0.5, 0, 0, 10, 0, 0, 0])
    collector = AttemptCollector(2, 2, initial, initial, 12)
    for i in range(1, 13):
        state = initial.copy()
        state[0] = (0.5 + i) % 12
        collector.append(state, state, np.zeros(2), i / 10, "mppi", 0, i - 1)
    assert collector.completed_lap and collector.goal_progress == 11.5


def test_only_incomplete_records_can_preserve_numerical_failure():
    collector = attempt("incomplete")
    collector.cartesian[-1][0] = np.nan
    record = collector.finish("incomplete", "nonfinite_state")
    assert np.isnan(record.cartesian[-1, 0]) and not record.physical_failure.any()
    with pytest.raises(ValueError, match="finite"):
        replace(
            record, outcome="failure", physical_failure=np.asarray([False, False, False, True])
        ).validate()


def test_empty_interrupted_attempt_and_transition_shape_validation():
    state = np.zeros(7)
    collector = AttemptCollector(1, 1, state, state, 12)
    state[0] = 100
    record = collector.finish("incomplete", "interrupted")
    assert record.cartesian[0, 0] == 0 and record.controls.shape == (0, 2)
    with pytest.raises(ValueError, match="T\\+1"):
        replace(record, cartesian=np.zeros((0, 7))).validate()


def test_dataset_is_local_and_records_configuration_without_importing_warmup(tmp_path):
    config = load_config(
        overrides={
            "simulation": {"seed": 7},
            "output": {"quick_load": True, "init_path": "warmup.npz", "resume": None},
        }
    )
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        dataset = TrajectoryDataset(directory, config, 12)
        assert load_trajectories(directory, "failure") == []
        dataset.add(attempt().finish("failure", "out_of_bounds"))
        assert dataset.counts == {"success": 0, "failure": 1, "incomplete": 0}
        context = json.loads((dataset.directory / "context.json").read_text())
        assert context["config"]["simulation"]["seed"] == 7
        assert context["warmup_source"].endswith("warmup.npz")
        assert len(load_trajectories(directory, "failure")) == 1
        assert load_trajectories(directory, "success") == []


def test_old_lap_files_are_not_accepted_as_new_trajectories(tmp_path):
    path = tmp_path / "laps.npz"
    np.savez(path, schema=np.asarray(1), **{"lap_0000/cartesian": np.zeros((3, 7))})
    with pytest.raises(ValueError, match="unsupported trajectory"):
        load_trajectory(path)
