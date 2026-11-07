import numpy as np
import pytest

from sit_lmpc.data.laps import Lap
from sit_lmpc.diagnostics.lap_metrics import LambdaStatistics, measured_boundary_metrics, summarize_lap


def make_lap(lateral):
    states = np.zeros((len(lateral), 7))
    states[:, 1] = lateral
    return Lap(
        states.copy(), states, np.zeros((len(states), 2)), np.arange(1, len(states) + 1) / 10
    )


def test_lambda_statistics_reports_per_term_max_and_mean_and_resets_between_attempts():
    statistics = LambdaStatistics()
    assert statistics.summary() == dict(steps=0, terms={})
    for values in ((2., 0.), (6., 4.), (0., 2.)):
        statistics.observe(values, ("hull", "boundary"))
    summary = statistics.summary()
    assert summary["steps"] == 3
    assert summary["terms"]["hull"] == dict(mean=8 / 3, max=6., steps=3)
    assert summary["terms"]["boundary"] == dict(mean=2., max=4., steps=3)
    statistics.clear()
    statistics.observe((0., 0.), ("hull", "boundary"))
    assert statistics.summary()["terms"]["hull"] == dict(mean=0., max=0., steps=1)
    assert summary["steps"] == 3 and summary["terms"]["hull"]["max"] == 6.


def test_lambda_statistics_uses_each_terms_active_steps_without_inventing_zero_weights():
    statistics = LambdaStatistics()
    statistics.observe((1., 0.), ("hull", "boundary"))
    statistics.observe((3., 2., 4., 6., 8.),
                       ("hull", "boundary", "chance_boundary", "chance_terminal_hull", "risk"))
    statistics.observe((5., 1., 2.), ("hull", "boundary", "risk"))
    summary = statistics.summary()
    assert summary["steps"] == 3
    assert summary["terms"]["hull"] == dict(mean=3., max=5., steps=3)
    assert summary["terms"]["risk"] == dict(mean=5., max=8., steps=2)
    assert summary["terms"]["chance_boundary"] == dict(mean=4., max=4., steps=1)
    assert summary["terms"]["chance_terminal_hull"] == dict(mean=6., max=6., steps=1)


def test_measured_excess_is_not_the_controller_penalty():
    lap = make_lap([0, -3.99, -4.2])
    metrics = summarize_lap(
        lap, 23, 4, [(0, 0)] * 3, completed=False, termination_reason="out_of_bounds"
    )
    assert metrics["boundary_max"] == 0
    assert metrics["actual_boundary_max"] == pytest.approx(0.2)
    assert metrics["actual_boundary_mean"] == pytest.approx(0.2 / 3)
    assert metrics["out_of_bounds_steps"] == 1
    assert metrics["max_abs_lateral_offset"] == pytest.approx(4.2)
    assert metrics["completed"] is False
    assert metrics["termination_reason"] == "out_of_bounds"
    assert metrics["duration"] == pytest.approx(0.3)
    assert metrics["lap_time"] is None
    assert metrics["mean_speed"] == 0


def test_valid_lap_time_exists_only_for_completed_laps_and_speed_uses_finite_samples():
    lap = make_lap([0, 0, 0])
    lap.cartesian[:, 3] = [7, np.nan, 9]
    metrics = summarize_lap(
        lap, 1, 4, [(0, 0)] * 3, completed=True, termination_reason="lap_completed"
    )
    assert metrics["lap_time"] == pytest.approx(0.3) and metrics["mean_speed"] == 8


def test_buffer_is_not_a_measured_track_exit():
    metrics = measured_boundary_metrics(make_lap([-4, 3.99, 0]), 4)
    assert metrics["actual_boundary_max"] == 0
    assert metrics["out_of_bounds_steps"] == 0


def test_nonfinite_measurements_are_reported_not_serialized_as_nan():
    metrics = measured_boundary_metrics(make_lap([np.nan, np.inf, 4.1]), 4)
    assert metrics["nonfinite_lateral_steps"] == 2
    assert metrics["actual_boundary_max"] == pytest.approx(0.1)
    assert measured_boundary_metrics(make_lap([np.nan]), 4)["actual_boundary_max"] is None
