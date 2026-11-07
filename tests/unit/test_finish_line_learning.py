"""Finish crossings stay complete in raw data, but continuous in learning data."""

from dataclasses import replace

import numpy as np
import pytest

from sit_lmpc.config.schema import PlannerConfig
from sit_lmpc.data.dynamics import trajectory_transitions
from sit_lmpc.data.trajectories import AttemptCollector
from sit_lmpc.planners.safe_set import SafeSet


def successful_attempt(number, start=0.2, end=0.4, *, reset=False):
    initial = np.asarray([start, 0, 0, 10, 0, 0, 0], dtype=float)
    cartesian = initial.copy()
    cartesian[0] += (number - 1) * 100 + (1 if reset else 0)
    collector = AttemptCollector(number, number, cartesian, initial, 100)
    for index, s in enumerate([1, 3, 6, 20, 40, 60, 80, 99, end]):
        frenet = initial.copy()
        frenet[0] = s
        state = frenet.copy()
        state[0] += (number - 1) * 100 + (100 if index == 8 else 0)
        control = np.asarray([number + index / 100, -index / 10])
        collector.append(state, frenet, control, (index + 1) / 10, "mppi", 0, index)
    return collector.finish("success", "lap_completed")


def safe_set():
    planner = replace(PlannerConfig(), safe_set_size=1, horizon=5, extension_seconds=1)
    return SafeSet(planner, 0.1, 100)


def test_learning_view_unwraps_terminal_state_without_changing_raw_trajectory():
    trajectory = successful_attempt(1)
    original = trajectory.frenet.copy()
    learning = trajectory.as_learning_lap(100)
    np.testing.assert_allclose(learning.frenet[:, 0], [1, 3, 6, 20, 40, 60, 80, 99, 100.4])
    np.testing.assert_allclose(np.diff(learning.frenet[:, 0]), np.diff(trajectory.progress[1:]))
    np.testing.assert_array_equal(learning.frenet[:, 1:], trajectory.frenet[1:, 1:])
    np.testing.assert_array_equal(learning.cartesian, trajectory.cartesian[1:])
    np.testing.assert_array_equal(learning.controls, trajectory.controls)
    np.testing.assert_array_equal(learning.times, trajectory.times[1:])
    np.testing.assert_array_equal(trajectory.frenet, original)
    assert trajectory.as_lap().frenet[-1, 0] == 0.4
    transitions = trajectory_transitions(trajectory, 0.01)
    assert len(transitions) == len(trajectory.controls)
    np.testing.assert_array_equal(transitions.next_states[-1], trajectory.cartesian[-1])
    np.testing.assert_array_equal(transitions.controls[-1], trajectory.controls[-1])


def test_single_transition_crossing_uses_initial_state_for_unwrapping():
    initial = np.asarray([99.5, 0, 0, 10, 0, 0, 0], dtype=float)
    collector = AttemptCollector(1, 1, initial, initial, 100)
    terminal = initial.copy()
    terminal[0] = 0.5
    collector.append(terminal, terminal, np.zeros(2), 0.1, "mppi", 0, 0)
    trajectory = collector.finish("success", "lap_completed")
    assert trajectory.as_learning_lap(100).frenet[0, 0] == 100.5
    assert trajectory.frenet[-1, 0] == 0.5


@pytest.mark.parametrize("with_trajectory", [False, True])
def test_extension_targets_and_progress_remain_continuous(with_trajectory):
    first = successful_attempt(1)
    second = successful_attempt(2, start=0.4, end=0.8)
    safe = safe_set()
    for trajectory in (first, second):
        safe.add_lap(trajectory.as_lap(), trajectory if with_trajectory else None)
    record = safe.frenet[0]
    np.testing.assert_allclose(record[:, 0], [1, 3, 6, 20, 40, 60, 80, 99, 100.4, 101, 103])
    np.testing.assert_allclose(record[:, -1], np.r_[first.times[1:] - 0.9, [0.1, 0.2]])
    assert np.all(np.diff(record[:, 0]) > 0)
    assert np.all(np.diff(record[:, -1]) > 0)
    # No full-lap positive target at the finish or terminal state near s=0.
    assert record[:, -1].max() == pytest.approx(0.2)
    assert not np.any(record[:, 0] < 0.5)
    assert not record.flags.writeable
    np.testing.assert_array_equal(safe.cartesian[0][:, -1], record[:, -1])
    np.testing.assert_array_equal(first.frenet[-1, 0], 0.4)
    if with_trajectory:
        np.testing.assert_array_equal(safe.controls[0][:8], first.controls[1:])
        np.testing.assert_array_equal(safe.controls[0][8], second.controls[0])
        np.testing.assert_array_equal(safe.controls[0][9:], second.controls[1:3])
        assert safe.control_masks[0].all()
        data = safe.training_data("state_action")
        np.testing.assert_array_equal(data[:, :5], record[:, :5])
        np.testing.assert_array_equal(data[:, 5:7], safe.controls[0])
        np.testing.assert_array_equal(data[:, -1], record[:, -1])
    else:
        assert not safe.control_masks[0].any()


def test_extension_uses_only_contiguous_starting_prefix():
    safe = safe_set()
    first = successful_attempt(1)
    second = successful_attempt(2, start=0.4, end=0.8)
    lap = second.as_lap()
    frenet = lap.frenet.copy()
    frenet[3, 0] = 4  # Re-entering the extension region later is not its prefix.
    safe.add_lap(first.as_lap())
    safe.add_lap(replace(lap, frenet=frenet))
    np.testing.assert_array_equal(safe.frenet[0][-2:, 0], [101, 103])


def test_unknown_action_after_reset_is_not_assigned_to_previous_endpoint():
    safe = safe_set()
    first = successful_attempt(1)
    second = successful_attempt(2, start=0.4, end=0.8, reset=True)
    safe.add_lap(first.as_lap(), first)
    safe.add_lap(second.as_lap(), second)
    assert not safe.control_masks[0][8]
    assert safe.control_masks[0][9:].all()


def test_extension_never_reuses_finish_endpoint_even_with_long_extension():
    safe = safe_set()
    safe.config = replace(safe.config, extension_seconds=100)
    first, second = successful_attempt(1), successful_attempt(2, start=0.4, end=0.8)
    safe.add_lap(first.as_lap(), first)
    safe.add_lap(second.as_lap(), second)
    assert len(safe.frenet[0]) == 9 + 8
    assert safe.frenet[0][-1, 0] == 199
