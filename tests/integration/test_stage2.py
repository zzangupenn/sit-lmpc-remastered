import json
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.data.initialization import save_initial
from sit_lmpc.data.trajectories import load_trajectories
from sit_lmpc.experiments import runner
from sit_lmpc.planners.common import PlannerResult


def setup_run(
    tmp_path, monkeypatch, *, failures=(), stop=None, quick_load=False, max_steps=30, max_laps=2
):
    captured = {"resets": [], "noise": [], "training": [], "updates": []}
    track = SimpleNamespace(s_frame_max=12, frenet_to_cartesian=lambda *args: (0, 0, 0))
    monkeypatch.setattr(runner.Track, "from_track_path", lambda path: track)

    class Simulator:
        def __init__(self, simulation, *args):
            self.steps = 0
            self.scale = np.asarray(simulation.action_scale)
            map_source = Path(args[0])
            self.run_directory = (
                map_source.parent.parent if map_source.parent.name == "map" else tmp_path / "run"
            )

        def reset(self, state):
            if self.steps:
                assert len(load_trajectories(self.run_directory, "failure")) == len(
                    captured["resets"]
                )
            captured["resets"].append(state.copy())
            self.state = state.copy()
            return (self.state.copy(), self.state.copy()), {"lap_counts": [0]}

        def snapshot(self):
            return {"steps": self.steps}

        def restore(self, state, bookkeeping):
            result = self.reset(state)
            self.steps = bookkeeping["steps"]
            frenet = result[0][1].copy()
            frenet[0] %= 12
            return (result[0][0], frenet), result[1]

        def step(self, action, noise):
            if stop == "interrupted" and self.steps == 3:
                raise KeyboardInterrupt
            self.steps += 1
            captured["noise"].append(noise.copy())
            self.state[0] += 1
            frenet = self.state.copy()
            frenet[0] %= 12
            collision = stop == "collision" and self.steps == 3
            if self.steps in failures:
                frenet[1] = -4.2
            if stop == "nonfinite_state" and self.steps == 3:
                self.state[0] = frenet[0] = np.nan
            terminal = collision or (stop == "simulator_terminated" and self.steps == 3)
            truncated = stop == "simulator_truncated" and self.steps == 3
            # Deliberately claim a finish on every step: only real full progress
            # may establish success, never reset/finish-line bookkeeping alone.
            return (
                (self.state.copy(), frenet),
                action * self.scale + noise,
                terminal,
                truncated,
                {"lap_counts": [self.steps], "collision": collision},
            )

        def render(self):
            pass

        def close(self):
            captured["simulator_closed"] = True

    class Tracking:
        def __init__(self, config, model, *args, randomness=None, **kwargs):
            self.actions = jnp.zeros((config.horizon, 2), dtype=model.params.dtype)
            self.randomness = randomness
            captured["nominal"] = self

        def update(self, state, speed):
            self.randomness.next_key()
            self.actions = jnp.ones_like(self.actions) * 0.2
            return PlannerResult(np.asarray([0.2, 0.1]), np.zeros((2, 7)))

    class Learning(Tracking):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if "state_action" in (kwargs.get("value_input_mode"), kwargs.get("risk_input_mode")):
                self.actions = jnp.pad(self.actions, ((0, 1), (0, 0)))
            captured["learned"] = self

        def precompile_hulls(self, emit):
            captured["precompiled"] = True

        def update(self, state, frenet, snapshot, safe_set, capture_debug=False):
            captured["updates"].append(
                (snapshot.generation, np.asarray(self.actions).copy(), len(safe_set.frenet))
            )
            return super().update(state, 10)

    class Trainer:
        def __init__(self, *args, randomness, **kwargs):
            self.generation, self.snapshot = 0, None
            self.randomness = randomness

        def start(self, data, epochs=None):
            captured["training"].append(data.copy())
            self.generation += 1
            self.snapshot = SimpleNamespace(generation=self.generation)

        def wait(self):
            pass

    monkeypatch.setattr(runner, "F1TenthSimulator", Simulator)
    monkeypatch.setattr(runner, "MPPI", Tracking)
    monkeypatch.setattr(runner, "LMPPI", Learning)
    monkeypatch.setattr(runner, "ValueTrainer", Trainer)
    config = load_config(
        overrides={
            "simulation": {
                "seed": 7,
                "max_steps": max_steps,
                "max_laps": max_laps,
            },
            "planner": {
                "horizon": 2,
                "samples": 8,
                "safe_set_size": 1,
                "sit_lmpc_mode": False,
                "belief": {"enabled": False},
                "risk_constraint": {"enabled": False},
            },
            "value": {"epochs": 1, "input_mode": "state"},
            "risk": {"enabled": False, "input_mode": "state"},
            "diagnostics": {
                "trace_steps": 0,
                "render": False,
                "plots": False,
                "keyboard_controls": False,
                "summary_plots": False,
            },
            "output": {
                "root": str(tmp_path),
                "exp_name": "test-resume",
                "directory": str(tmp_path / "run"),
                "quick_load": quick_load,
                "init_path": str(tmp_path / "warmup.npz"),
                "resume": None,
            },
        }
    )
    if quick_load:
        safe = runner.SafeSet(config.planner, 0.1, 12)
        # Use the real numeric warm-up snapshot/loader, not a mocked import.
        warmup = np.zeros((12, 8))
        warmup[:, 0], warmup[:, 3] = np.arange(12), 10
        warmup[:, -1] = np.arange(12) / 10 - 1.2
        safe.frenet, safe.cartesian = [warmup], [warmup.copy()]
        save_initial(
            config.output.init_path,
            safe,
            np.asarray([0, 0, 0.1, 10, 0.2, 0.03, 0.04]),
            np.zeros((2, 2)),
            config,
        )
    return config, captured


def test_multiple_failures_are_saved_before_reset_and_success_training_is_clean(
    tmp_path, monkeypatch
):
    config, captured = setup_run(tmp_path, monkeypatch, failures=(3, 30), max_steps=42, max_laps=5)
    result = runner.run(config)
    failures = load_trajectories(result.directory, "failure")
    successes = load_trajectories(result.directory, "success")
    assert (
        result.status == "lap_limit"
        and result.laps == 3
        and result.failures == 2
        and result.trials == 5
    )
    assert [t.attempt_id for t in failures] == [1, 4]
    assert [t.attempt_id for t in successes] == [2, 3, 5]
    for index, failed in enumerate(failures):
        np.testing.assert_array_equal(captured["resets"][index + 1], failed.cartesian[0])
        assert failed.cartesian.shape == (4, 7) and failed.frenet[-1, 1] == -4.2
        assert not failed.physical_failure[:-1].any() and failed.physical_failure[-1]
        assert not failed.goal.any()
    assert len(captured["noise"]) == 42
    assert not np.array_equal(captured["noise"][0], captured["noise"][3])
    assert len(captured["training"]) == 2
    assert all((data[:, 1] == 0).all() for data in captured["training"])
    assert len(captured["updates"]) == 15 and all(update[0] == 1 for update in captured["updates"])
    np.testing.assert_array_equal(captured["updates"][3][1], np.zeros((2, 2)))
    assert load_trajectories(result.directory, "incomplete") == []
    summary = json.loads((result.directory / "summary.json").read_text())
    assert summary["attempts"] == 5 and summary["failed_attempts"] == 2


def test_disabled_risk_avoids_endpoint_padding_and_failure_reset(tmp_path, monkeypatch):
    from dataclasses import replace

    config, captured = setup_run(tmp_path, monkeypatch, quick_load=True, failures=(3,), max_steps=5)
    config = replace(config, risk=replace(config.risk, enabled=True, input_mode="state_action"))
    runner.run(config)
    restored = captured["updates"][0][1]
    assert restored.shape == (config.planner.horizon, 2)
    from sit_lmpc.data.initialization import load_initial

    np.testing.assert_array_equal(
        restored, load_initial(config.output.init_path, config).actions
    )
    np.testing.assert_array_equal(captured["updates"][3][1], np.zeros_like(restored))


@pytest.mark.parametrize(
    "failures,max_laps,expected_successes,expected_steps",
    [
        ((3, 6), 3, 1, 18),
        ((3, 6, 9, 12), 3, 0, 9),
    ],
)
def test_max_laps_counts_successes_and_failures_and_stops_without_an_extra_retry(
    tmp_path, monkeypatch, failures, max_laps, expected_successes, expected_steps
):
    config, captured = setup_run(
        tmp_path, monkeypatch, failures=failures, max_steps=50, max_laps=max_laps
    )
    result = runner.run(config)
    assert result.status == "lap_limit" and result.trials == max_laps
    assert result.laps == expected_successes and result.failures == max_laps - expected_successes
    assert len(captured["noise"]) == expected_steps and len(captured["resets"]) == 3
    assert load_trajectories(result.directory, "incomplete") == []
    summary = json.loads((result.directory / "summary.json").read_text())
    assert summary["trials"] == max_laps and summary["successful_laps"] == expected_successes
    assert summary["partial_lap"] is None


@pytest.mark.parametrize("reason", ["out_of_bounds", "collision"])
def test_sit_lmpc_mode_disables_bank_and_saves_then_stops_on_first_failure(
    tmp_path, monkeypatch, reason
):
    from dataclasses import replace

    config, captured = setup_run(
        tmp_path,
        monkeypatch,
        failures=(3,) if reason == "out_of_bounds" else (),
        stop="collision" if reason == "collision" else None,
        quick_load=True,
        max_steps=30,
        max_laps=20,
    )
    config = replace(
        config,
        planner=replace(config.planner, sit_lmpc_mode=True),
    )
    monkeypatch.setattr(runner, "interactive_available", lambda config: True)
    monkeypatch.setattr(
        runner,
        "ModelComparisons",
        lambda *args: pytest.fail("legacy mode constructed a model bank"),
    )
    result = runner.run(config)
    assert result.status == reason and result.trials == result.failures == 1 and result.laps == 0
    assert len(captured["resets"]) == 1 and len(captured["noise"]) == 3
    assert (
        result.learned_steps == 3 and result.training_generations == 1 and captured["precompiled"]
    )
    assert load_trajectories(result.directory, "failure")[0].termination_reason == reason
    assert load_trajectories(result.directory, "incomplete") == []
    assert not list(result.directory.glob("model_comparison*"))
    saved = load_config(result.directory / "config.yaml")
    assert saved.planner.sit_lmpc_mode and saved.simulation.friction == config.simulation.friction
    assert saved.planner.friction == config.planner.friction


def test_sit_lmpc_toggle_preserves_control_and_rng_before_first_failure(tmp_path, monkeypatch):
    from dataclasses import replace

    records = []
    for mode in (False, True):
        config, _ = setup_run(tmp_path / str(mode), monkeypatch, failures=(3,), max_steps=3)
        config = replace(config, planner=replace(config.planner, sit_lmpc_mode=mode))
        result = runner.run(config)
        records.append(load_trajectories(result.directory, "failure")[0])
    for field in (
        "cartesian",
        "frenet",
        "controls",
        "times",
        "progress",
        "controllers",
        "value_generations",
    ):
        np.testing.assert_array_equal(getattr(records[0], field), getattr(records[1], field))


def test_lap_summary_mode_refreshes_on_failures_as_well_as_successes(tmp_path, monkeypatch):
    from dataclasses import replace

    config, _ = setup_run(tmp_path, monkeypatch, failures=(3,), max_steps=30, max_laps=2)
    config = replace(
        config, diagnostics=replace(config.diagnostics, keyboard_controls=True, summary_plots=True)
    )
    calls = []

    class Keyboard(runner.KeyboardControls):
        def start(self):
            self.command("2")

    class Plots:
        def __init__(self, *args):
            pass

        def summaries(self, laps, metrics, *args, show=False, pause=False, keep_paused=None):
            calls.append((len(laps), [item["outcome"] for item in metrics], show))
            if show:
                assert pause and keep_paused()

        def pump(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "KeyboardControls", Keyboard)
    monkeypatch.setattr("sit_lmpc.diagnostics.restored_plots.MatplotlibPlots", Plots)
    result = runner.run(config)
    assert result.trials == 2
    assert calls == [
        (0, ["failure"], True),
        (1, ["failure", "success"], True),
        (1, ["failure", "success"], False),
    ]


def test_failed_finish_line_crossing_is_not_success_and_step_budget_does_not_reset(
    tmp_path, monkeypatch
):
    config, captured = setup_run(tmp_path, monkeypatch, failures=(12,), max_steps=12)
    result = runner.run(config)
    assert result.status == "step_limit" and result.laps == 0 and result.failures == 1
    failed = load_trajectories(result.directory, "failure")[0]
    assert failed.progress[-1] == 12 and not failed.goal.any()
    assert failed.controls.shape == (12, 2) and len(captured["noise"]) == 12
    assert len(captured["resets"]) == 2 and captured["training"] == []


@pytest.mark.parametrize(
    "reason",
    ["interrupted", "simulator_truncated", "simulator_terminated", "nonfinite_state", "step_limit"],
)
def test_administrative_and_numerical_stops_are_incomplete_not_failures(
    tmp_path, monkeypatch, reason
):
    config, captured = setup_run(
        tmp_path, monkeypatch, stop=reason, max_steps=3 if reason == "step_limit" else 10
    )
    result = runner.run(config)
    assert result.status == reason and result.laps == result.failures == 0
    record = load_trajectories(result.directory, "incomplete")[0]
    assert record.termination_reason == reason and record.controls.shape == (3, 2)
    assert not record.physical_failure.any() and not record.goal.any()
    assert len(captured["resets"]) == 1 and captured["training"] == []
    assert (
        load_trajectories(result.directory, "success")
        == load_trajectories(result.directory, "failure")
        == []
    )


def test_collision_is_a_physical_failure_and_retries(tmp_path, monkeypatch):
    config, captured = setup_run(tmp_path, monkeypatch, stop="collision", max_steps=5)
    result = runner.run(config)
    failed = load_trajectories(result.directory, "failure")[0]
    assert result.status == "step_limit" and result.failures == 1
    assert failed.termination_reason == "collision" and failed.frenet[-1, 1] == 0
    assert len(captured["resets"]) == 2
    assert len(load_trajectories(result.directory, "incomplete")) == 1


def test_warmup_quick_load_works_and_learning_survives_failure_reset(tmp_path, monkeypatch):
    config, captured = setup_run(tmp_path, monkeypatch, quick_load=True, failures=(3,), max_steps=5)
    result = runner.run(config)
    assert result.learned_steps == 5 and result.failures == 1 and result.training_generations == 1
    assert len(captured["training"]) == 1
    assert all(update[0] == 1 and update[2] == 1 for update in captured["updates"])
    np.testing.assert_array_equal(captured["updates"][3][1], np.zeros((2, 2)))
    # Imported warm-up initializes learning, not this run's outcome collections.
    assert load_trajectories(result.directory, "success") == []
    assert len(load_trajectories(result.directory, "failure")) == 1
    np.testing.assert_array_equal(captured["resets"][1], captured["resets"][0])


def test_summary_plot_error_does_not_lose_failure_or_incomplete_data(tmp_path, monkeypatch):
    config, captured = setup_run(tmp_path, monkeypatch, failures=(3,), max_steps=5)
    from dataclasses import replace

    config = replace(config, diagnostics=replace(config.diagnostics, summary_plots=True))

    class BrokenPlots:
        def __init__(self, *args):
            pass

        def pump(self):
            pass

        def summaries(self, *args):
            raise RuntimeError("test graphics failure")

        def close(self):
            pass

    monkeypatch.setattr("sit_lmpc.diagnostics.restored_plots.MatplotlibPlots", BrokenPlots)
    result = runner.run(config)
    assert result.status == "step_limit" and result.failures == 1 and captured["simulator_closed"]
    assert (
        len(load_trajectories(result.directory, "failure"))
        == len(load_trajectories(result.directory, "incomplete"))
        == 1
    )
    assert "test graphics failure" in (result.directory / "events.jsonl").read_text()
