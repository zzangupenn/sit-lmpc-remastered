from dataclasses import replace

import pytest

from sit_lmpc.experiments import runner
from test_stage2 import setup_run


@pytest.mark.parametrize("failure_phase", [None, "track", "model", "trainer"])
def test_keyboard_starts_before_initialization_and_closes_on_startup_failure(tmp_path, monkeypatch, failure_phase):
    config, _ = setup_run(tmp_path, monkeypatch, max_steps=1)
    config = replace(config, diagnostics=replace(config.diagnostics, keyboard_controls=True))
    calls, controls = [], []

    class Keyboard(runner.KeyboardControls):
        def start(self):
            calls.append("listener started")
            controls.append(self)
            self.command("6")  # A selection made before model/training setup.

        def close(self):
            calls.append("listener closed")
            super().close()

    original_track = runner.Track.from_track_path
    original_model = runner.build_vehicle_dynamics
    original_trainer = runner.ValueTrainer

    def track(*args):
        assert calls == ["listener started"]
        assert controls[0].safe_set_cost_mode()
        calls.append("track")
        if failure_phase == "track":
            raise RuntimeError("test startup failure")
        return original_track(*args)

    def model(*args):
        assert controls[0].safe_set_cost_mode()
        calls.append("model")
        if failure_phase == "model":
            raise RuntimeError("test startup failure")
        return original_model(*args)

    def trainer(*args, **kwargs):
        assert controls[0].safe_set_cost_mode()
        calls.append("trainer")
        if failure_phase == "trainer":
            raise RuntimeError("test startup failure")
        # The selection survived startup. Release it for the non-GUI test run.
        controls[0].command("0")
        return original_trainer(*args, **kwargs)

    monkeypatch.setattr(runner, "KeyboardControls", Keyboard)
    monkeypatch.setattr(runner, "interactive_available", lambda diagnostics: False)
    monkeypatch.setattr(runner.Track, "from_track_path", track)
    monkeypatch.setattr(runner, "build_vehicle_dynamics", model)
    monkeypatch.setattr(runner, "ValueTrainer", trainer)
    if failure_phase is None:
        assert runner.run(config).status == "step_limit"
        assert calls == ["listener started", "track", "model", "trainer", "listener closed"]
    else:
        with pytest.raises(RuntimeError, match="test startup failure"):
            runner.run(config)
        assert calls[-1] == "listener closed" and calls.count("listener closed") == 1
