from dataclasses import replace
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
from sit_lmpc.dynamics import STModel, step
from sit_lmpc.diagnostics.model_comparison import ModelComparisons
from sit_lmpc.experiments import runner
from sit_lmpc.track import Track, map_path


@pytest.mark.parametrize("noise_std", [0, 0.001])
def test_true_reference_and_recorded_controls_match_gym(tmp_path, noise_std):
    config = load_config()
    sim = replace(config.simulation, friction=0.5, seed=19, initial_speed=12)
    path = map_path("custom1")
    track = Track.from_track_path(path)
    pose = track.frenet_to_cartesian(0, 0, 0)
    simulator = F1TenthSimulator(sim, path, False, vehicle_parameters(0.5))
    belief = STModel.from_vehicle(vehicle_parameters(0.6), sim)
    truth = STModel.from_vehicle(vehicle_parameters(0.5), sim)
    comparisons = ModelComparisons(belief, truth, (0.3, 0.4, 0.7, 0.8), 10)
    rng = np.random.default_rng(4)
    try:
        (state, _), _ = simulator.reset(np.asarray([pose[0], pose[1], 0.1, 12, pose[2], 0, 0]))
        initial = state.copy()
        comparisons.warmup(state)
        for i in range(10):
            (next_state, _), control, _, _, _ = simulator.step(
                np.asarray([0.3, 0.1]), rng.normal(0, noise_std, 2)
            )
            comparisons.observe(state, control / np.asarray(sim.action_scale), next_state)
            state = next_state
        result = comparisons.recorded(9)
        np.testing.assert_array_equal(result.actual[0], initial)
        np.testing.assert_allclose(result.true_reference, result.actual, rtol=2e-5, atol=2e-4)
        assert np.max(np.abs(result.candidates[0] - result.actual)) > 1e-3
        assert result.actions.shape == (10, 2)
        assert result.evaluation_seconds >= 0
        assert not list(tmp_path.glob("model_comparison*"))
    finally:
        simulator.close()


@pytest.mark.parametrize("stop_after_first", [False, True])
@pytest.mark.parametrize("sit_lmpc_mode", [False, True])
def test_pause_comparison_is_generated_before_each_applied_step_in_warmup(
    tmp_path, monkeypatch, stop_after_first, sit_lmpc_mode
):
    import threading

    calls, observed, controls, closed_at, safe_views = [], [], [], [], []
    safe_states = np.zeros((8, 8))
    safe_states[:, :2] = np.column_stack((np.arange(8), np.full(8, 0.2)))

    class SafeSet(runner.SafeSet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # One retained record: still tracking, not enough laps for LMPPI.
            self.frenet = [safe_states.copy()]
            self.cartesian = [safe_states.copy()]

    class Keyboard(runner.KeyboardControls):
        def start(self):
            self.command("5")
            controls.append(self)

    class Simulator(runner.F1TenthSimulator):
        def render(self):
            if controls and controls[0].comparison_mode():
                assert calls, "Option 5 paused without opening a comparison plot"
            return super().render()

        def step(self, action, noise):
            observed.append(np.asarray(action).copy())
            return super().step(action, noise)

    class Plots:
        def __init__(self, *args):
            pass

        def model_comparison(self, comparison, track, width, show=False, *, safe_set_states):
            np.testing.assert_array_equal(safe_set_states, safe_states)
            safe_views.append(comparison.kind)
            if comparison.kind == "recorded":
                return None
            assert len(observed) == comparison.step
            assert comparison.kind == "planned" and comparison.actual.shape == (1, 7)
            assert comparison.frictions == (
                (config.planner.friction,) if sit_lmpc_mode else config.planner.model_bank.frictions
            )
            assert show
            return comparison

        def wait_for_interaction(self, comparison, keep_paused):
            index = comparison.step
            assert keep_paused()
            calls.append(index)
            if stop_after_first:
                # Model the global keyboard listener, but require Qt closure
                # to happen in the runner's main-thread pause callback.
                listener = threading.Thread(target=controls[0].command, args=("0",))
                listener.start()
                listener.join(timeout=1)
                assert not listener.is_alive()
                assert not keep_paused()

        def pump(self):
            pass

        def close(self):
            assert threading.current_thread() is threading.main_thread()
            closed_at.append(len(observed))

    monkeypatch.setattr(runner, "KeyboardControls", Keyboard)
    monkeypatch.setattr(runner, "interactive_available", lambda config: config.keyboard_controls)
    monkeypatch.setattr(runner, "F1TenthSimulator", Simulator)
    monkeypatch.setattr(runner, "SafeSet", SafeSet)
    monkeypatch.setattr("sit_lmpc.diagnostics.restored_plots.MatplotlibPlots", Plots)
    config = load_config(
        overrides={
            "simulation": {"seed": 7, "max_steps": 3},
            "planner": {
                "samples": 8, "horizon": 2, "sit_lmpc_mode": sit_lmpc_mode,
                "safe_set_size": 4, "dynamics": {"backend": "st"},
                "probabilistic_safety": {"enabled": False},
                "risk_constraint": {"enabled": False},
            },
            "diagnostics": {
                "keyboard_controls": True,
                "summary_plots": False,
                "plots": False,
                "render": False,
            },
            "output": {"directory": str(tmp_path / "run"), "quick_load": False, "resume": None},
        }
    )
    assert runner.run(config, tracking_only=True).status == "step_limit"
    assert calls == ([0] if stop_after_first else [0, 1, 2]) and len(observed) == 3
    assert closed_at == ([0, 3] if stop_after_first else [3])
    assert safe_views == ["planned"] * len(calls) + ["recorded"]


@pytest.mark.parametrize("sit_lmpc_mode", [False, True])
def test_diagnostics_toggle_preserves_actions_states_and_live_predictor(
    tmp_path, monkeypatch, sit_lmpc_mode
):
    pytest.importorskip("matplotlib")
    live_updates = []

    class Live:
        def update(self, state, predicted, action):
            live_updates.append((state.copy(), predicted.copy(), action.copy()))

        def close(self):
            pass

    monkeypatch.setattr("sit_lmpc.diagnostics.plots.LivePlots", Live)
    # No key is selected: enabling the listener must not alter the controller.
    monkeypatch.setattr(runner.KeyboardControls, "start", lambda self: None)

    class Plots:
        def __init__(self, *args):
            pass

        def model_comparison(self, *args):
            pass

        def pump(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "interactive_available", lambda config: config.keyboard_controls)
    monkeypatch.setattr("sit_lmpc.diagnostics.restored_plots.MatplotlibPlots", Plots)
    results = []
    for enabled in (False, True):
        config = load_config(
            overrides={
                "simulation": {"seed": 7, "max_steps": 4},
                "planner": {"samples": 8, "horizon": 3, "sit_lmpc_mode": sit_lmpc_mode},
                "diagnostics": {
                    "keyboard_controls": enabled,
                    "summary_plots": False,
                    "plots": True,
                    "render": False,
                    "plot_interval": 1,
                },
                "output": {
                    "directory": str(tmp_path / str(enabled)),
                    "quick_load": False,
                    "resume": None,
                },
            }
        )
        result = runner.run(config, tracking_only=True)
        with np.load(result.directory / "partial_lap.npz", allow_pickle=False) as data:
            results.append({name: data[name].copy() for name in data.files})
        assert not list(result.directory.glob("model_comparison*"))
    for name in results[0]:
        np.testing.assert_array_equal(results[0][name], results[1][name])
    for first, second in zip(live_updates[:4], live_updates[4:], strict=True):
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)


def plot_comparison(tmp_path, recorded=False):
    sim = load_config().simulation
    belief = STModel.from_vehicle(vehicle_parameters(0.6), sim)
    truth = STModel.from_vehicle(vehicle_parameters(0.5), sim)
    comparisons = ModelComparisons(belief, truth, (0.3, 0.4, 0.7, 0.8), 3)
    state = np.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=np.float32)
    actions = np.tile(np.asarray([0.3, 0.1]), (3, 1))
    if not recorded:
        return comparisons.planned(state, actions, 4)
    for action in actions:
        next_state = np.asarray(step(truth, state, action))
        comparisons.observe(state, action, next_state)
        state = next_state
    return comparisons.recorded(4)


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("interactive", [False, True])
def test_comparison_figure_labels_truth_and_reuses_one_nonlive_window(
    tmp_path, recorded, interactive
):
    pytest.importorskip("matplotlib")
    from matplotlib.layout_engine import TightLayoutEngine
    from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots

    comparison = plot_comparison(tmp_path, recorded)
    plots = MatplotlibPlots(tmp_path, interactive=False)
    # Test interactive display through Agg without opening a desktop window.
    plots.interactive = interactive
    track = Track.from_track_path(map_path("custom1"))
    try:
        safe_states = np.vstack((comparison.belief, comparison.true_reference))
        figure = plots.model_comparison(comparison, track, 4, safe_set_states=safe_states)
        labels = figure.axes[0].get_legend_handles_labels()[1]
        assert "Safe-set states" in labels
        np.testing.assert_array_equal(
            figure.axes[0].collections[0].get_offsets(), safe_states[:, :2]
        )
        assert len(figure.axes) == 1  # Bank/ST comparison needs no BNN zoom.
        assert len([label for label in labels if label.startswith("Candidate")]) == 4
        assert any(label.startswith("Belief ST") for label in labels)
        truth_line = next(
            line for line in figure.axes[0].lines
            if line.get_label().startswith("Ground-truth ST trajectory")
        )
        assert truth_line.get_color() == "tab:orange"
        assert truth_line.get_linewidth() == 0.9
        np.testing.assert_array_equal(truth_line.get_xydata(), comparison.true_reference[:, :2])
        assert ("Measured Gym trajectory" in labels) is recorded
        assert any("future unobserved" in label for label in labels) is not recorded
        assert isinstance(figure.get_layout_engine(), TightLayoutEngine)
        assert plots.model_comparison(comparison, track, 4) is figure
        assert len(plots.figures) == 1 and not plots.summary_figures
        assert not list(tmp_path.glob("model_comparison*"))
    finally:
        plots.close()


@pytest.mark.parametrize("display", [False, True])
def test_noninteractive_run_without_insight_never_constructs_bank_or_exports_comparison(
    tmp_path, monkeypatch, display
):
    monkeypatch.setattr(
        runner, "ModelComparisons", lambda *args: pytest.fail("noninteractive bank constructed")
    )
    if display:
        monkeypatch.setenv("DISPLAY", ":test")
    else:
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    config = load_config(
        overrides={
            "simulation": {"seed": 7, "max_steps": 3},
            "planner": {"samples": 8, "horizon": 2},
            "diagnostics": {
                "render": False,
                "plots": False,
                "keyboard_controls": False,
                "summary_plots": True,
                "trace_steps": 0,  # Historical option 5 otherwise needs a comparison bank.
            },
            "output": {"directory": str(tmp_path / "run"), "quick_load": False, "resume": None},
        }
    )
    result = runner.run(config, tracking_only=True)
    assert result.status == "step_limit"
    assert not list(result.directory.glob("model_comparison*"))
    assert (result.directory / "summary.png").is_file()


def test_qtagg_click_in_comparison_releases_pause(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt6")
    script = """
import sys, time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
app = QApplication([])
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
from matplotlib.backend_bases import MouseEvent
p = MatplotlibPlots(sys.argv[1], interactive=True)
comparison = p.plt.figure()
p.figures.append(comparison)
other = p.plt.figure()
QTimer.singleShot(75, lambda: comparison.canvas.callbacks.process('button_press_event', MouseEvent('button_press_event', comparison.canvas, 10, 10, button=1)))
started = time.perf_counter()
p.wait_for_interaction(comparison)
assert 0.03 <= time.perf_counter() - started < 3
p.close()
assert p.plt.fignum_exists(other.number)
p.plt.close(other)
print('Comparison click released one controller step')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("sit_lmpc_mode", [False, True])
@pytest.mark.parametrize("nf_density,option", [(False, "5"), (True, "5"), (True, "6")])
def test_headless_cli_keeps_real_qt_keyboard_comparison_and_continuation(tmp_path, sit_lmpc_mode, nf_density, option):
    import os
    import subprocess
    import sys

    pytest.importorskip("matplotlib")
    pytest.importorskip("PyQt6")
    script = """
import sys, json
from pathlib import Path
import yaml
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from matplotlib.backend_bases import KeyEvent, MouseEvent
from sit_lmpc.cli.racecar import arguments
from sit_lmpc.config.schema import ExperimentConfig
from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
from sit_lmpc.experiments import runner

app = QApplication([])
folder = Path(sys.argv[1])
settings = ExperimentConfig().as_dict()
settings['simulation'].update(seed_mode='fixed', seed=7, max_steps=2)
settings['planner'].update(samples=8, horizon=2, sit_lmpc_mode=bool(int(sys.argv[2])))
nf = bool(int(sys.argv[3]))
option = sys.argv[4]
if nf:
    settings['planner']['safe_set_query'] = 'nf_density'
settings['planner']['belief']['enabled'] = False
settings['risk']['enabled'] = False
settings['diagnostics'].update(render=True, plots=False, keyboard_controls=True, summary_plots=False)
settings['output'].update(directory=str(folder / 'run'), quick_load=False, resume=None)
preset = folder / 'config.yaml'
preset.write_text(yaml.safe_dump(settings))
sys.argv = ['sit-racecar', '--config', str(preset), '--headless']
_, config = arguments()
assert not config.diagnostics.render and config.diagnostics.keyboard_controls
from sit_lmpc.experiments.runtime import configure
configure(config, cpu=True)

calls = []
class Keyboard(runner.KeyboardControls):
    def start(self):
        self.command(option)

class Plots(MatplotlibPlots):
    def __init__(self, directory, interactive):
        assert interactive
        super().__init__(directory, interactive)

    def wait_for_interaction(self, figure, keep_paused=lambda: True):
        assert keep_paused()
        figures = figure if isinstance(figure, list) else [figure]
        assert len(figures) == 1
        if option == '6':
            assert 'Safe-set inspection' in figures[0]._suptitle.get_text()
            assert len(figures[0].axes) == 5  # Two cost panels, cutoff, mapping and a colorbar.
            assert figures[0].axes[2].texts[0].get_text() == 'NF calibration is not ready yet.'
            assert self.density_calibration_figure is None
        else:
            assert figures[0].axes[0].get_xlabel() == 'x (m)'
            assert self.density_calibration_figure is None
        calls.append('pause')
        target = figures[-1]  # The new NF window owns the same step/resume controls.
        if option == '6' and len(calls) == 1:
            QTimer.singleShot(50, lambda: target.canvas.callbacks.process(
                'button_press_event', MouseEvent('button_press_event', target.canvas, 10, 10, button=1)))
        else:
            QTimer.singleShot(50, lambda: target.canvas.callbacks.process(
                'key_press_event', KeyEvent('key_press_event', target.canvas, key='0')))
        super().wait_for_interaction(figure, keep_paused)
        if option == '6' and len(calls) == 1:
            assert keep_paused() and all(self.plt.fignum_exists(f.number) for f in figures)
        else:
            assert not keep_paused()
            assert not any(self.plt.fignum_exists(f.number) for f in figures)

runner.KeyboardControls = Keyboard
# Qt offscreen is a real GUI backend; no X server is needed for this test.
runner.interactive_available = lambda diagnostics: diagnostics.keyboard_controls
import sit_lmpc.diagnostics.restored_plots as restored
restored.MatplotlibPlots = Plots
result = runner.run(config, tracking_only=not nf)
assert result.status == 'step_limit'
assert json.loads((result.directory / 'summary.json').read_text())['applied_steps'] == 2
assert calls == ['pause'] * (2 if option == '6' else 1)
assert not list(result.directory.glob('model_comparison*'))
assert not (result.directory / 'summary.png').exists()
print('Headless simulator retained Qt keyboard plot, pause, closure, and continuation')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(int(sit_lmpc_mode)), str(int(nf_density)), option],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
