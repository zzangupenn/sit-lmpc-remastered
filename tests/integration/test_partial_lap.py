import json
import sys

import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.data.laps import load_laps
from sit_lmpc.data.laps import Lap, save_laps
from sit_lmpc.data.trajectories import load_trajectories
from sit_lmpc.diagnostics.lap_metrics import summarize_lap
from sit_lmpc.experiments import runner
from sit_lmpc.planners.common import PlannerResult


@pytest.mark.parametrize("exit_step", [3, 12, 15])
def test_exit_sample_and_partial_metrics_are_saved_without_training(
    tmp_path, monkeypatch, exit_step
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        runner.Track,
        "from_track_path",
        lambda path: SimpleNamespace(s_frame_max=12, frenet_to_cartesian=lambda *args: (0, 0, 0)),
    )

    class Simulator:
        def __init__(self, *args):
            self.steps = 0

        def reset(self, state):
            self.state = state.copy()
            return (self.state.copy(), self.state.copy()), {}

        def step(self, action, noise):
            self.steps += 1
            frenet = self.state.copy()
            frenet[0] = self.steps % 12
            frenet[1] = -4.2 if self.steps == exit_step else 0
            return (
                (self.state.copy(), frenet),
                np.zeros(2),
                False,
                False,
                {"lap_counts": [self.steps // 12]},
            )

        def render(self):
            pass

        def close(self):
            pass

    class Tracking:
        def __init__(self, *args, **kwargs):
            self.actions = np.zeros((2, 2))

        def update(self, *args):
            # Prediction says no violation even when the actual vehicle exits.
            return PlannerResult(np.zeros(2), np.zeros((2, 7)))

    monkeypatch.setattr(runner, "F1TenthSimulator", Simulator)
    monkeypatch.setattr(runner, "MPPI", Tracking)
    config = load_config(
        overrides={
            "simulation": {
                "seed_mode": "fixed",
                "seed": 7,
                "max_steps": exit_step,
            },
            "planner": {"horizon": 2, "samples": 8, "sit_lmpc_mode": False},
            "diagnostics": {
                "render": False,
                "plots": False,
                "keyboard_controls": False,
                "summary_plots": False,
            },
            "output": {"directory": str(tmp_path / "run"), "quick_load": False, "resume": None},
        }
    )
    result = runner.run(config, tracking_only=True)
    completed = (exit_step - 1) // 12
    assert result.status == "step_limit" and result.laps == completed and result.failures == 1
    assert result.training_generations == 0
    assert len(load_laps(result.directory / "laps.npz")) == completed
    partial = load_trajectories(result.directory, "failure")[0]
    assert partial.frenet[-1, 1] == pytest.approx(-4.2)
    assert len(partial.times) == exit_step - completed * 12 + 1
    records = json.loads((result.directory / "lap_metrics.json").read_text())
    assert len(records) == completed + 1
    failed = records[-1]
    assert failed["completed"] is False
    assert failed["lap"] == completed + 1
    assert failed["termination_reason"] == "out_of_bounds"
    assert failed["actual_boundary_max"] == pytest.approx(0.2)
    assert failed["boundary_max"] == 0
    summary = json.loads((result.directory / "summary.json").read_text())
    assert summary["partial_lap"] is None and summary["failed_attempts"] == 1
    events = [
        json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["event"] == "attempt" and event["outcome"] == "failure" for event in events)


@pytest.mark.parametrize("has_partial_metrics", [False, True])
def test_replay_includes_old_and_new_partial_attempts_without_rewriting_records(
    tmp_path, monkeypatch, has_partial_metrics
):
    from sit_lmpc.cli import replay

    config = load_config(overrides={"output": {"quick_load": False}})
    import yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config.as_dict()))
    states = np.zeros((3, 7))
    completed = Lap(states.copy(), states.copy(), np.zeros((3, 2)), np.arange(1, 4) / 10)
    states[-1, 1] = -4.2
    partial = Lap(states.copy(), states.copy(), np.zeros((3, 2)), np.arange(1, 4) / 10)
    save_laps(tmp_path / "laps.npz", [completed])
    save_laps(tmp_path / "partial_lap.npz", [partial])
    (tmp_path / "summary.json").write_text(json.dumps({"status": "out_of_bounds", "laps": 1}))
    records = [{"lap": 1, "boundary_max": 0, "boundary_mean": 0}]
    if has_partial_metrics:
        records.append(
            summarize_lap(
                partial, 2, 4, [(0, 0)] * 3, completed=False, termination_reason="out_of_bounds"
            )
        )
    (tmp_path / "lap_metrics.json").write_text(json.dumps(records))
    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    calls = []

    class Plots:
        def __init__(self, *args, **kwargs):
            pass

        def summaries(self, laps, metrics, track, half_width):
            calls.append(metrics)
            assert len(laps) == 1

        def close(self):
            pass

    monkeypatch.setattr("sit_lmpc.diagnostics.restored_plots.MatplotlibPlots", Plots)
    monkeypatch.setattr("sit_lmpc.experiments.runtime.configure", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["sit-replay", "--run", str(tmp_path), "--headless"])
    replay.main()
    assert len(calls[0]) == 2
    assert calls[0][-1]["completed"] is False
    assert calls[0][-1]["termination_reason"] == "out_of_bounds"
    assert calls[0][-1]["actual_boundary_max"] == pytest.approx(0.2)
    if not has_partial_metrics:
        assert "boundary_max" not in calls[0][-1]
    assert all(path.read_bytes() == value for path, value in originals.items())
