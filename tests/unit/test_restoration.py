from dataclasses import replace
import sys
import jax
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.data.initialization import (
    InitialSnapshot,
    load_initial,
    save_initial,
    restore_safe_set,
)
from sit_lmpc.diagnostics.keyboard import KeyboardControls, interactive_available
from sit_lmpc.experiments.randomness import ExperimentRandomness, resolve_seed
from sit_lmpc.planners.safe_set import SafeSet
from sit_lmpc.training.value import TrainingRandomness


def test_fresh_seed_resolution_and_fixed_replay(monkeypatch):
    seeds = iter((123, 456))
    monkeypatch.setattr("sit_lmpc.experiments.randomness.secrets.randbits", lambda _: next(seeds))
    config = load_config()
    assert config.simulation.seed is None
    assert resolve_seed(config).simulation.seed == 123
    assert resolve_seed(config).simulation.seed == 456
    fixed = load_config(overrides={"simulation": {"seed": 42}})
    assert fixed.simulation.seed_mode == "fixed"
    assert resolve_seed(fixed).simulation.seed == 42


def test_truncated_noise_is_bounded_without_clipping_atoms():
    a, b = ExperimentRandomness(42), ExperimentRandomness(42)
    noise = a.state_noise(0.2, 100000)
    np.testing.assert_array_equal(noise, b.state_noise(0.2, 100000))
    assert (np.abs(noise) < 0.6).all()
    assert abs(noise.mean()) < 0.002
    np.testing.assert_array_equal(a.state_noise(0), np.zeros(7))


def test_host_and_jax_streams_share_one_owner():
    a, b = ExperimentRandomness(13), ExperimentRandomness(13)
    for owner in (a, b):
        safe = SafeSet(load_config().planner, 0.1, 100, randomness=owner)
        assert safe.rng is owner.host
    np.testing.assert_array_equal(
        jax.random.key_data(a.next_key()), jax.random.key_data(b.next_key())
    )
    first, second = a.next_key(), a.next_key()
    assert not np.array_equal(jax.random.key_data(first), jax.random.key_data(second))


def test_training_randomness_preallocated_and_advances_master_stream():
    config = replace(load_config().value, epochs=2, boundary_samples=3)
    a, b = ExperimentRandomness(1), ExperimentRandomness(1)
    plan = TrainingRandomness.allocate(a.host, 7, config)
    duplicate = TrainingRandomness.allocate(b.host, 7, config)
    np.testing.assert_array_equal(plan.boundary, duplicate.boundary)
    for permutation in plan.permutations:
        np.testing.assert_array_equal(np.sort(permutation), np.arange(10))
    np.testing.assert_array_equal(a.state_noise(0.1), b.state_noise(0.1))
    other = TrainingRandomness.allocate(a.host, 7, config)
    assert not np.array_equal(plan.boundary, other.boundary)


@pytest.mark.parametrize("enabled", [True, False])
def test_keyboard_controls_gate_debug_pause_and_summaries(enabled):
    config = replace(load_config().diagnostics, keyboard_controls=enabled)
    controls = KeyboardControls(config)
    controls.command("1")
    assert controls.poll() == (enabled, False)
    assert controls.should_pause() is enabled
    controls.command("2")
    assert controls.poll() == (False, enabled)
    assert controls.poll() == (False, enabled)
    assert not controls.should_pause()
    controls.command("3")
    assert controls.risk_mode() is enabled
    assert controls.poll() == (False, False)
    assert controls.should_pause() is enabled
    controls.command("4")
    assert controls.validation_mode() is enabled
    assert controls.validation_selection() == (enabled, int(enabled))
    controls.command("4")
    assert controls.validation_selection() == (enabled, 2 * int(enabled))
    assert not controls.should_pause()
    controls.command("5")
    assert controls.comparison_mode() is enabled
    assert controls.should_pause() is enabled
    controls.command("6")
    assert controls.safe_set_cost_mode() is enabled
    assert not controls.comparison_mode()
    assert controls.should_pause() is enabled
    controls.command("0")
    assert controls.poll() == (False, False)
    assert not controls.risk_mode()
    assert not controls.validation_mode()
    assert not controls.comparison_mode()
    assert not controls.safe_set_cost_mode()
    assert controls.consume_close_request() is enabled
    assert not controls.consume_close_request()
    controls.command("alt_r")
    assert controls.should_pause() is enabled
    controls.command("1")
    controls.command("0")
    assert controls.poll() == (False, False)


def test_window_close_request_is_latched_across_keyboard_mode_changes():
    controls = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
    controls.command("0")
    controls.command("1")
    assert controls.should_pause()
    assert controls.consume_close_request()
    assert not controls.consume_close_request()
    assert controls.should_pause()


@pytest.mark.parametrize("mode", ["1", "2", "3", "4", "5", "6"])
def test_keyboard_suspension_clears_views_ignores_shortcuts_and_resumes(mode):
    messages = []
    controls = KeyboardControls(
        replace(load_config().diagnostics, keyboard_controls=True),
        lambda event, **fields: messages.append((event, fields)),
    )
    controls.command(mode)
    controls.consume_close_request()
    controls.command("toggle_listening")
    assert controls.suspended and not controls.should_pause()
    assert controls.poll() == (False, False)
    assert not controls.risk_mode() and not controls.comparison_mode() and not controls.validation_mode()
    assert not controls.safe_set_cost_mode()
    assert controls.consume_close_request()
    request = controls.validation_selection()[1]
    for key in ("0", "1", "2", "3", "4", "5", "6", "alt_r"):
        controls.command(key)
    assert not controls.should_pause() and not controls.consume_close_request()
    assert controls.validation_selection() == (False, request)
    controls.command("toggle_listening")
    assert not controls.suspended and not controls.should_pause()  # No old pause restored.
    controls.command("5")
    assert controls.comparison_mode() and controls.should_pause()
    listening = [fields for event, fields in messages if event == "keyboard_listening"]
    assert [fields["enabled"] for fields in listening] == [False, True]
    received = [fields for event, fields in messages if event == "keyboard_received"]
    assert [fields["accepted"] for fields in received] == [True] + [False] * 8 + [True]


def test_disabled_keyboard_cannot_be_enabled_by_suspend_chord():
    controls = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=False))
    controls.command("toggle_listening")
    assert not controls.suspended and not controls.should_pause() and not controls.consume_close_request()


def test_switching_number_modes_closes_previous_plot_but_reselecting_does_not():
    controls = KeyboardControls(replace(load_config().diagnostics, keyboard_controls=True))
    controls.command("1")
    assert not controls.consume_close_request()
    controls.command("1")
    assert not controls.consume_close_request()
    for key in ("2", "3", "4", "5", "6", "1"):
        controls.command(key)
        assert controls.consume_close_request()
        assert not controls.consume_close_request()
        controls.command(key)
        assert not controls.consume_close_request()
    controls.command("alt_r")
    assert controls.consume_close_request()
    assert not controls.consume_close_request()
    assert not controls.should_pause()


def test_disabled_keyboard_does_not_import_or_start_listener(monkeypatch):
    config = replace(load_config().diagnostics, keyboard_controls=False)
    monkeypatch.setitem(sys.modules, "pynput", None)
    controls = KeyboardControls(config)
    controls.start()
    controls.close()
    assert controls.listener is None


@pytest.mark.parametrize('tracking_mode', [False, True])
def test_keyboard_listener_shortcuts_and_cleanup(monkeypatch, tracking_mode):
    from types import SimpleNamespace

    calls = []

    ctrl, alt, alt_right = object(), object(), object()

    class HotKey:
        def __init__(self, keys, callback):
            self.keys, self.callback, self.held = set(keys), callback, set()

        @staticmethod
        def parse(chord):
            assert chord in ("<ctrl>+<alt>+k", "<ctrl>+<alt>+r")
            return (ctrl, alt, chord[-1])

        def press(self, key):
            if key in self.keys and key not in self.held:
                self.held.add(key)
                if self.held == self.keys:
                    self.callback()

        def release(self, key):
            self.held.discard(key)

    class Listener:
        def __init__(self, on_press, on_release):
            self.on_press, self.on_release = on_press, on_release

        def canonical(self, key):
            if getattr(key, "char", None) is not None:
                return key.char.lower()
            return alt if key is alt_right else key

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

        def join(self, timeout):
            calls.append("join")

    module = SimpleNamespace(
        keyboard=SimpleNamespace(Listener=Listener, HotKey=HotKey,
                                 Key=SimpleNamespace(alt_r=alt_right, ctrl=ctrl, alt=alt,
                                                     left=object(), right=object()))
    )
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setitem(sys.modules, "pynput", module)
    messages = []
    controls = KeyboardControls(
        replace(load_config().diagnostics, keyboard_controls=True, render=False),
        lambda event, **fields: messages.append((event, fields)),
        extra_keys=('r',) if tracking_mode else (),
        extra_key_chords={'r': '<ctrl>+<alt>+r'} if tracking_mode else None,
    )
    controls.start()
    assert messages[0][0] == "keyboard_started"
    assert "5: sampled dynamics trajectories / model comparison" in messages[0][1]["shortcuts"]
    assert "Ctrl+Alt+K: suspend/resume shortcuts" in messages[0][1]["shortcuts"]
    controls.listener.on_release(alt_right)
    assert controls.poll()[0]
    controls.listener.on_release(SimpleNamespace(char="0"))
    assert not controls.poll()[0]
    controls.listener.on_release(SimpleNamespace(char="2"))
    assert controls.poll()[1]
    controls.listener.on_release(SimpleNamespace(char="3"))
    assert controls.risk_mode() and controls.should_pause()
    controls.listener.on_release(SimpleNamespace(char="4"))
    assert controls.validation_mode() and not controls.should_pause()
    controls.listener.on_release(SimpleNamespace(char="5"))
    assert controls.comparison_mode() and controls.should_pause()
    controls.listener.on_release(SimpleNamespace(char="6"))
    assert controls.safe_set_cost_mode() and controls.should_pause()
    assert not controls.comparison_mode()
    listener = controls.listener
    for key in (ctrl, alt_right, SimpleNamespace(char="K")):
        listener.on_press(key)
    assert controls.suspended and not controls.should_pause()
    listener.on_press(SimpleNamespace(char="K"))  # Held-key repeat does not resume.
    assert controls.suspended
    for key in (SimpleNamespace(char="K"), ctrl, alt_right):
        listener.on_release(key)
    assert controls.suspended and not controls.should_pause()  # No right Alt debug toggle.
    listener.on_release(SimpleNamespace(char="5"))
    assert not controls.comparison_mode()
    for key in (ctrl, alt, SimpleNamespace(char="k")):
        listener.on_press(key)
    for key in (SimpleNamespace(char="k"), alt, ctrl):
        listener.on_release(key)
    assert not controls.suspended and not controls.should_pause()
    listener.on_press(ctrl)
    listener.on_release(SimpleNamespace(char="5"))  # Modified numbers aren't debug commands.
    listener.on_release(ctrl)
    assert not controls.comparison_mode()
    listener.on_release(SimpleNamespace(char="5"))
    assert controls.comparison_mode() and controls.should_pause()
    if tracking_mode:
        # Typing ordinary R/r in an editor or chat must not switch simulation mode.
        for letter in ('r', 'R'):
            listener.on_press(SimpleNamespace(char=letter))
            listener.on_release(SimpleNamespace(char=letter))
        assert controls.take_extra_commands() == []
        view = controls.view()
        for key in (ctrl, alt_right, SimpleNamespace(char='R')):
            listener.on_press(key)
        listener.on_press(SimpleNamespace(char='R'))  # Held-key repeat is one command.
        for key in (ctrl, alt_right, SimpleNamespace(char='R')):
            listener.on_release(key)  # Releasing modifiers before R must not queue it twice.
        assert controls.take_extra_commands() == ['r']
        assert controls.view() == view  # Right Alt in the chord is not a debug toggle.
        controls.toggle_listening()
        for key in (ctrl, alt, SimpleNamespace(char='r')):
            listener.on_press(key)
        for key in (SimpleNamespace(char='r'), alt, ctrl):
            listener.on_release(key)
        assert controls.take_extra_commands() == []
    controls.close()
    assert calls == ["start", "stop", "join"]
    assert controls.listener is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_debug_proposal_returns_same_candidates_without_new_sampling(dtype):
    import jax.numpy as jnp
    from sit_lmpc.planners.lmppi import propose, debug_reward_terms

    config = replace(load_config().planner, samples=7, horizon=2, lambda_count=2)
    rng = np.random.default_rng(8)
    previous = jnp.zeros((2, 2), dtype=dtype)
    da = jnp.asarray(rng.uniform(-0.2, 0.2, (7, 2, 2)), dtype=dtype)
    states = jnp.asarray(rng.normal(size=(7, 2, 7)), dtype=dtype)
    values = jnp.arange(14, dtype=dtype)
    distances = jnp.asarray(rng.uniform(size=7), dtype=dtype)
    lambdas = jnp.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=dtype)
    args = (previous, da, states, values, distances, lambdas, config)
    regular = propose(*args)
    terms = debug_reward_terms(*args)
    debug = propose(*args)
    np.testing.assert_array_equal(regular, debug)
    assert terms["reward"].shape == (4, 7, 2)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("live_plots", [True, False])
def test_headless_only_disables_simulator_render(tmp_path, monkeypatch, enabled, live_plots):
    from sit_lmpc.cli.racecar import arguments
    import yaml

    preset = tmp_path / "config.yaml"
    preset.write_text(
        yaml.safe_dump({"output": {"resume": None}, "diagnostics": {
            "render": True, "keyboard_controls": enabled, "plots": live_plots,
            "summary_plots": True,
        }})
    )
    monkeypatch.setattr(sys, "argv", ["sit-racecar", "--config", str(preset), "--headless"])
    _, config = arguments()
    assert not config.diagnostics.render
    assert config.diagnostics.keyboard_controls is enabled
    assert config.diagnostics.plots is live_plots
    assert config.diagnostics.summary_plots
    monkeypatch.setenv("DISPLAY", ":test")
    assert interactive_available(config.diagnostics) is enabled


@pytest.mark.parametrize("display", [None, "DISPLAY", "WAYLAND_DISPLAY"])
@pytest.mark.parametrize("enabled", [True, False])
def test_keyboard_views_need_display_but_not_simulator_render(monkeypatch, display, enabled):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if display:
        monkeypatch.setenv(display, ":test")
    config = replace(load_config().diagnostics, render=False, keyboard_controls=enabled)
    assert interactive_available(config) is (enabled and display is not None)


@pytest.mark.parametrize(
    "settings", [{"keyboard_controls": "false"}, {"keyboard_controls": 1}, {"render": None}]
)
def test_invalid_toggle_rejected(settings):
    with pytest.raises(ValueError):
        load_config(overrides={"diagnostics": settings})


def test_keyboard_schema_default_is_off_and_user_preset_is_respected():
    from importlib.resources import files
    from sit_lmpc.config.schema import DiagnosticsConfig

    assert DiagnosticsConfig().keyboard_controls is False
    expected = load_config(files("sit_lmpc") / "config" / "default.yaml")
    for config in (load_config(), load_config(files("sit_lmpc") / "config" / "default.yaml")):
        assert config.diagnostics.keyboard_controls == expected.diagnostics.keyboard_controls
        assert "keyboard_pause" not in config.as_dict()["diagnostics"]


@pytest.mark.parametrize("enabled", [True, False])
def test_old_config_and_overrides_ignore_retired_keyboard_switches(tmp_path, enabled):
    import yaml

    path = tmp_path / "config.yaml"
    retired = dict.fromkeys(("keyboard_pause", "debug_plots", "model_comparison", "interactive"), not enabled)
    path.write_text(yaml.safe_dump({"diagnostics": {"keyboard_controls": enabled, **retired}}))
    for config in (load_config(path), load_config(path, {"diagnostics": retired})):
        assert config.diagnostics.keyboard_controls is enabled
        assert not set(retired) & set(config.as_dict()["diagnostics"])
        controls = KeyboardControls(config.diagnostics)
        for key in ("1", "3", "5"):
            controls.command(key)
            assert controls.should_pause() is enabled


def fixture_snapshot():
    config = load_config(
        overrides={"planner": {"safe_set_size": 1, "horizon": 3}, "value": {"input_mode": "state"}}
    )
    safe = SafeSet(config.planner, 0.1, 100)
    values = np.arange(80, dtype=float).reshape(10, 8) / 100
    safe.frenet = [values]
    safe.cartesian = [values.copy()]
    state = np.array([0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0])
    actions = np.zeros((3, 2))
    return config, safe, state, actions


def test_native_init_roundtrip_and_compatibility(tmp_path):
    config, safe, state, actions = fixture_snapshot()
    path = tmp_path / "init.npz"
    save_initial(path, safe, state, actions, config)
    with np.load(path, allow_pickle=False) as payload:
        assert all(payload[key].dtype != object for key in payload.files)
    snapshot = load_initial(path, config)
    np.testing.assert_array_equal(snapshot.state, state)
    np.testing.assert_array_equal(snapshot.actions, actions)
    target = SafeSet(config.planner, 0.1, 100)
    restore_safe_set(target, snapshot)
    assert target.ready and target.previous is None
    assert not target.frenet[0].flags.writeable
    with pytest.raises(ValueError, match="planner.friction"):
        load_initial(path, replace(config, planner=replace(config.planner, friction=0.8)))
    with pytest.raises(FileExistsError):
        save_initial(path, safe, state, actions, config)


def test_legacy_requires_explicit_trust_and_reads_utilitysuite_format(tmp_path):
    config, safe, state, actions = fixture_snapshot()
    fields = {
        "safe_set": safe.frenet,
        "safe_set_xy": safe.cartesian,
        "init_state": state,
        "a_opt_records": [[actions]],
    }
    path = tmp_path / "init.npz"
    np.savez(path, **{key: {key: value} for key, value in fields.items()})
    with pytest.raises(ValueError, match="pickle"):
        load_initial(path, config)
    snapshot = load_initial(path, config, trust_legacy=True)
    np.testing.assert_array_equal(snapshot.state, state)
    np.testing.assert_array_equal(snapshot.actions, actions)
    assert snapshot.metadata["schema"] == "trusted-legacy"


def test_quick_load_requires_path_for_new_experiments_but_resume_uses_checkpoint():
    with pytest.raises(ValueError, match="init_path"):
        load_config(overrides={"output": {"quick_load": True, "init_path": None, "resume": None}})
    config = load_config(
        overrides={"output": {"quick_load": True, "init_path": None, "resume": "checkpoint"}}
    )
    assert config.output.resume == "checkpoint"
