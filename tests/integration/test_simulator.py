import numpy as np
import pytest
from sit_lmpc.config import load_config
from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
from sit_lmpc.track import Track, map_path


def test_reset_step_close_in_headless_mode():
    config = load_config()
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    simulator = F1TenthSimulator(
        config.simulation, map_path("custom1"), False, vehicle_parameters(0.6)
    )
    try:
        (cartesian, frenet), _ = simulator.reset(np.array([pose[0], pose[1], 0, 7, pose[2], 0, 0]))
        assert cartesian.shape == frenet.shape == (7,)
        (next_state, next_frenet), control, terminated, truncated, _ = simulator.step(
            np.zeros(2), np.zeros(2)
        )
        assert np.isfinite(next_state).all() and np.isfinite(next_frenet).all()
        assert control.shape == (2,)
    finally:
        simulator.close()


def test_retry_reset_restores_exact_full_state_counters_and_keeps_rng():
    config = load_config()
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    initial = np.asarray(
        [pose[0], pose[1], 0.1, 12, pose[2], 0.03, 0.04], dtype=config.simulation.dtype
    )
    simulator = F1TenthSimulator(
        config.simulation, map_path("custom1"), False, vehicle_parameters(0.5)
    )
    try:
        (state, _), _ = simulator.reset(initial)
        np.testing.assert_array_equal(state, initial)
        raw = simulator.env.unwrapped
        simulator.step(np.asarray([0.2, 0.1]), np.zeros(2))
        raw.lap_counts[:] = 4
        raw.lap_times[:] = 20
        raw.sim.collisions[:] = 1
        raw.sim.agents[0].in_collision = True
        rng = raw.sim.agents[0].scan_rng
        rng.random()
        rng_state = rng.bit_generator.state
        global_state = np.random.get_state()
        (reset_state, _), info = simulator.reset(initial)
        np.testing.assert_array_equal(reset_state, initial)
        assert raw.sim_time == 0 and info["sim_time"] == 0
        assert not raw.lap_counts.any() and not raw.lap_times.any() and not raw.collisions.any()
        assert not raw.sim.agents[0].in_collision and raw.agents_prev_s[0] is None
        assert raw.sim.agents[0].scan_rng is rng and rng.bit_generator.state == rng_state
        after = np.random.get_state()
        assert after[0] == global_state[0] and after[2:] == global_state[2:]
        np.testing.assert_array_equal(after[1], global_state[1])
        (_, _), _, terminated, _, _ = simulator.step(np.zeros(2), np.zeros(2))
        assert not terminated and raw.sim_time == pytest.approx(config.simulation.timestep)
    finally:
        simulator.close()


def test_trial_boundary_restore_recovers_vehicle_and_lap_bookkeeping():
    config = load_config(
        overrides={
            "simulation": {
                "seed_mode": "fixed",
                "seed": 7,
                "friction": 0.5,
                "max_laps": 2,
                "max_steps": 20,
            },
            "planner": {
                "friction": 0.6,
                "sit_lmpc_mode": False,
                "risk_constraint": {"enabled": False},
            },
            "risk": {"enabled": False, "input_mode": "state"},
            "diagnostics": {
                "render": False,
                "plots": False,
                "keyboard_controls": False,
            },
        }
    )
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    initial = np.asarray([pose[0], pose[1], 0.0, 8, pose[2], 0, 0])
    simulator = F1TenthSimulator(
        config.simulation, map_path("custom1"), False, vehicle_parameters(0.5)
    )
    try:
        simulator.reset(initial)
        (state, frenet), _, _, _, _ = simulator.step(np.zeros(2), np.zeros(2))
        bookkeeping = simulator.snapshot()
        simulator.step(np.zeros(2), np.zeros(2))
        (restored, restored_frenet), _ = simulator.restore(state, bookkeeping)
        np.testing.assert_allclose(restored, state, atol=1e-6)
        np.testing.assert_allclose(restored_frenet, frenet, atol=1e-6)
        assert simulator.snapshot() == bookkeeping
    finally:
        simulator.close()


def test_runner_uses_planner_friction_for_both_controllers_and_gym_friction_for_simulator(
    tmp_path, monkeypatch
):
    import json
    from sit_lmpc.experiments import runner
    from sit_lmpc.diagnostics import plots
    from sit_lmpc.dynamics.st import PARAMETER_FIELDS

    captured = {}

    class Tracking(runner.MPPI):
        def __init__(self, config, model, *args, **kwargs):
            captured["mppi"] = float(model.params[PARAMETER_FIELDS.index("mu")])
            super().__init__(config, model, *args, **kwargs)

    class Learning(runner.LMPPI):
        def __init__(self, config, model, *args, **kwargs):
            captured["lmppi"] = float(model.params[PARAMETER_FIELDS.index("mu")])
            super().__init__(config, model, *args, **kwargs)

    class Simulator(runner.F1TenthSimulator):
        def __init__(self, *args):
            super().__init__(*args)
            captured["gym"] = self.env.unwrapped.sim.agents[0].params["mu"]

    monkeypatch.setattr(runner, "MPPI", Tracking)
    monkeypatch.setattr(runner, "LMPPI", Learning)
    monkeypatch.setattr(runner, "F1TenthSimulator", Simulator)
    monkeypatch.setattr(plots, "save_summary", lambda *args: None)
    config = load_config(
        overrides={
            "simulation": {"friction": 0.6, "max_steps": 2, "seed_mode": "fixed", "seed": 11},
            "planner": {"friction": 0.8, "samples": 8, "horizon": 2,
                        "belief": {"enabled": False}},
            "diagnostics": {
                "render": False,
                "plots": False,
                "keyboard_controls": False,
            },
            "output": {"directory": str(tmp_path / "run"), "quick_load": False, "resume": None},
        }
    )
    result = runner.run(config, tracking_only=True)
    assert result.status == "step_limit"
    assert captured == pytest.approx({"mppi": 0.8, "lmppi": 0.8, "gym": 0.6})
    saved = load_config(result.directory / "config.yaml")
    assert saved.simulation.friction == 0.6 and saved.planner.friction == 0.8
    start = next(event for line in (result.directory / "events.jsonl").read_text().splitlines() if (event := json.loads(line))["event"] == "start")
    assert start["simulator_friction"] == 0.6 and start["predictor_friction"] == 0.8


def test_disabled_keyboard_debug_does_not_block_control(tmp_path, monkeypatch):
    from sit_lmpc.experiments import runner

    class DebugKeyboard(runner.KeyboardControls):
        def start(self):
            self.command("1")

    monkeypatch.setattr(runner, "KeyboardControls", DebugKeyboard)
    config = load_config(
        overrides={
            "simulation": {"seed": 7, "max_steps": 2},
            "planner": {"samples": 8, "horizon": 2},
            "diagnostics": {
                "render": False,
                "plots": False,
                "summary_plots": False,
                "keyboard_controls": False,
            },
            "output": {"directory": str(tmp_path / "run"), "quick_load": False, "resume": None},
        }
    )
    result = runner.run(config, tracking_only=True)
    assert result.status == "step_limit"
    assert "keyboard_paused" not in (result.directory / "events.jsonl").read_text()
