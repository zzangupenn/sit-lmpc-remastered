"""Numerical equivalence, bounded dispatch, speculative RNG, and enabled diagnostics."""

from copy import deepcopy
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import ExperimentConfig, PlannerConfig, load_config
from sit_lmpc.experiments import runner
from sit_lmpc.planners.common import to_frenet
from sit_lmpc.planners.mppi import MPPI, iteration, tracking_control
from sit_lmpc.planners.safe_set import SafeSet, PrecompiledHullEvaluator, build_hull, hull_distance
from sit_lmpc.track import Track, map_path
from sit_lmpc.training.features import inference_snapshot, InferenceState
from sit_lmpc.dynamics.gaussian_bnn import sample_coefficients


@pytest.mark.parametrize("changes", [
    {"safe_set_selection": "bad"}, {"safe_set_window_margin": -1},
    {"safe_set_window_margin": float("nan")}, {"safe_set_window_margin": True},
    {"projection_block_size": 0}, {"projection_block_size": True},
])
def test_optimization_settings_validate(changes):
    with pytest.raises(ValueError):
        load_config(overrides={"planner": changes})


def test_safe_arrays_cache_and_selection_preserve_rng_and_original_algorithm():
    config = replace(PlannerConfig(), safe_set_max_points=90, hull_selection_points=12)
    safe = SafeSet(config, 0.1, 100, seed=17)
    rng = np.random.default_rng(9)
    safe.frenet = [rng.normal(size=(60, 8)), rng.normal(size=(60, 8))]
    safe.cartesian = [row.copy() for row in safe.frenet]
    cached = safe.arrays()
    assert safe.arrays()[0] is cached[0] and safe.arrays()[1] is cached[1]
    reference_rng = deepcopy(safe.rng)
    f, c = np.concatenate(safe.frenet), np.concatenate(safe.cartesian)
    cap = reference_rng.choice(len(f), 90, replace=False)
    f, c = f[cap], c[cap]
    lo, hi = safe.window((-0.2, 0.2), 10.)
    indices = np.flatnonzero((f[:, 0] > lo) & (f[:, 0] < hi))
    indices = reference_rng.choice(indices, 12, replace=False)
    selected, cartesian = safe.select_bounds((-0.2, 0.2), 10.)
    np.testing.assert_array_equal(selected, f[indices])
    np.testing.assert_array_equal(cartesian, c[indices])
    assert safe.rng.bit_generator.state == reference_rng.bit_generator.state
    safe.frenet[0] = safe.frenet[0].copy()
    assert safe.arrays()[0] is not cached[0]


def test_speculation_only_consumes_rng_when_committed_and_does_not_mutate_records():
    safe = SafeSet(replace(PlannerConfig(), hull_selection_points=10), 0.1, 100, seed=7)
    rng = np.random.default_rng(8)
    safe.frenet = [rng.normal(size=(100, 8))]
    safe.cartesian = [safe.frenet[0].copy()]
    state = deepcopy(safe.rng.bit_generator.state)
    selected, reference, hull, after, signature = safe.speculate((-10, 10), jnp.float64)
    assert safe.rng.bit_generator.state == state and signature == safe.signature
    expected, expected_c = safe.select_window(-10, 10)
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(reference, expected_c)
    assert safe.rng.bit_generator.state == after
    assert np.isfinite(hull.equations).all()
    before = deepcopy(safe.rng.bit_generator.state)
    with pytest.raises(ValueError):
        safe.speculate((100, 101), jnp.float64)
    assert safe.rng.bit_generator.state == before


@pytest.mark.parametrize("hull_query", ["facets", "implicit"])
def test_predicted_window_miss_rebuilds_exact_geometry_without_extra_rng(monkeypatch, hull_query):
    from types import SimpleNamespace
    from sit_lmpc.planners import lmppi

    base = ExperimentConfig()
    config = replace(base.planner, samples=8, horizon=3, lambda_count=2, hull_precompile=False,
                     hull_selection_points=12, safe_set_selection="predicted_window",
                     safe_set_window_margin=0, safe_set_query=hull_query)
    _, model = runner.build_vehicle_dynamics(base)
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = np.asarray([pose[0], pose[1], 0, 10, pose[2], 0, 0])
    frenet = np.asarray([0, 0, 0, 10, 0, 0, 0])
    rows = np.random.default_rng(7).normal(size=(300, 8))
    rows[:, 0] = np.linspace(-5, 15, 300)
    rows[:, 3] += 10
    exact_setting = replace(config, safe_set_selection="exact")
    predicted = SafeSet(config, .1, track.s_frame_max, seed=11)
    exact = SafeSet(exact_setting, .1, track.s_frame_max, seed=11)
    for safe in (predicted, exact):
        safe.frenet, safe.cartesian = [rows], [rows.copy()]
    monkeypatch.setattr(lmppi, "predict_value", lambda snapshot, states, *a: jnp.zeros(len(states)))
    value = SimpleNamespace(input_mode="state", generation=1)
    approximate = lmppi.LMPPI(config, model, track, seed=17).update(state, frenet, value, predicted)
    reference = lmppi.LMPPI(exact_setting, model, track, seed=17).update(state, frenet, value, exact)
    assert approximate.metrics["safe_set_speculation"] == "window_miss"
    np.testing.assert_allclose(approximate.action, reference.action, atol=1e-7)
    assert predicted.rng.bit_generator.state == exact.rng.bit_generator.state
    assert approximate.metrics["hull_facets"] == reference.metrics["hull_facets"]


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float32, 2e-5), (jnp.float64, 1e-11)])
def test_particle_hull_scan_preserves_all_facets_and_does_not_dispatch_per_query(monkeypatch, dtype, tolerance):
    rng = np.random.default_rng(9)
    scorer = PrecompiledHullEvaluator(7, 3, bucket=10, limit=40, dtype=dtype)
    scorer.precompile()
    scorer.precompile_many((31,))
    compiled = dict(scorer.compiled_many)
    assert len(compiled) == 3
    monkeypatch.setattr(scorer, "evaluate", lambda *a: pytest.fail("host query loop"))
    queries = jnp.asarray(rng.normal(size=(31, 5)), dtype)
    for n in (12, 35, 45):
        hull = build_hull(rng.normal(size=(n, 5)), 10, dtype)
        np.testing.assert_allclose(scorer.evaluate_many(queries, hull), hull_distance(queries, hull),
                                   atol=tolerance, rtol=tolerance)
    assert scorer.compiled_many == compiled


def test_projection_chunks_preserve_global_lap_lift_and_remove_padding():
    track = Track.from_track_path(map_path("custom1"))
    progress = np.asarray([track.s_frame_max - 2, track.s_frame_max - 1, 0, 1, 2])
    poses = np.asarray([track.frenet_to_cartesian(s, .2, .01) for s in progress])
    states = np.zeros((5, 7))
    states[:, :2], states[:, 4], states[:, 3] = poses[:, :2], poses[:, 2], 10
    indices = jnp.asarray(track.get_s_search_range(progress[0], 400))
    expected = to_frenet(track, jnp.asarray(states), indices, progress[0], 0)
    actual = to_frenet(track, jnp.asarray(states), indices, progress[0], 3)
    np.testing.assert_allclose(actual, expected, atol=1e-8)


@pytest.mark.parametrize("iterations", [1, 3])
def test_tracking_scan_matches_original_iterations_and_warmup_preserves_state(iterations):
    base = ExperimentConfig()
    config = replace(base.planner, samples=8, horizon=3, iterations=iterations)
    _, model = runner.build_vehicle_dynamics(base)
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = jnp.asarray([pose[0], pose[1], 0, 10, pose[2], 0, 0], dtype=model.params.dtype)
    controller = MPPI(config, model, track)
    initial_key = jax.random.key_data(controller.key).copy()
    controller.warmup(state, 10)
    np.testing.assert_array_equal(jax.random.key_data(controller.key), initial_key)
    np.testing.assert_array_equal(controller.actions, 0)
    keys = jax.random.split(jax.random.key(7), iterations)
    reference = controller.reference(state, 10)
    expected = controller.actions
    for key in keys:
        expected, prediction, _ = iteration(model, state, expected, reference, key, config)
    optimal, trajectory, actual_reference, _ = tracking_control(
        model, state, controller.actions, controller.waypoints, controller.diff,
        controller.distances, 10, keys, config,
    )
    np.testing.assert_allclose(optimal, expected, atol=2e-6)
    np.testing.assert_allclose(trajectory, prediction, atol=2e-6)
    np.testing.assert_allclose(actual_reference, reference, atol=1e-9)


def test_inference_only_snapshots_match_training_snapshots_without_optimizer_inputs():
    from sit_lmpc.data.normalization import Normalization
    from sit_lmpc.training.value import ValueSnapshot, initial_state as value_state, predict_value
    from sit_lmpc.training.risk import RiskSnapshot, initial_state as risk_state, predict_risk

    base = ExperimentConfig()
    normalization = Normalization(jnp.zeros(6), jnp.ones(6))
    value = ValueSnapshot(value_state(replace(base.value, hidden_dims=8, layers=1), jax.random.key(1)),
                          normalization, 1)
    risk = RiskSnapshot(risk_state(replace(base.risk, hidden_dims=8, layers=1), jax.random.key(2)),
                         normalization, 1)
    states = jnp.ones((8, 7)) * .3
    for snapshot, predict in ((value, lambda s: predict_value(s, states, jax.random.key(3), 2)),
                              (risk, lambda s: predict_risk(s, states))):
        reduced = inference_snapshot(snapshot)
        assert isinstance(reduced.state, InferenceState) and not hasattr(reduced.state, "opt_state")
        assert hasattr(snapshot.state, "opt_state") and reduced.state.params is snapshot.state.params
        np.testing.assert_array_equal(predict(snapshot), predict(reduced))
        assert inference_snapshot(reduced) is reduced


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_compiled_sampler_keeps_original_rounding_boundaries(dtype):
    rng = np.random.default_rng(23)
    mean = jnp.asarray(rng.normal(size=96) * 20, dtype)
    factor = jnp.asarray(rng.normal(size=(96, 96)) * .01, dtype)
    key = jax.random.key(7)
    normal = jax.random.normal(key, (64, 96), dtype=dtype)
    product = jnp.matmul(normal, factor.T, precision=jax.lax.Precision.HIGHEST)
    expected = mean + product
    np.testing.assert_array_equal(sample_coefficients(mean, factor, key, 64), expected)


@pytest.mark.parametrize("trace,live,expected", [(True, False, 3), (False, True, 2),
                                                (True, True, 3), (False, False, 0)])
def test_enabled_trace_and_live_keep_applied_control_predictions(tmp_path, monkeypatch, trace, live, expected):
    calls, displayed = [], []
    original = runner.step

    def predicted(*args):
        calls.append(tuple(np.asarray(x).copy() if hasattr(x, "shape") else x for x in args[1:]))
        return original(*args)

    class Live:
        def update(self, state, prediction, action):
            assert prediction is not None and np.isfinite(prediction).all()
            displayed.append(np.asarray(prediction).copy())

        def close(self):
            pass

    monkeypatch.setattr(runner, "step", predicted)
    monkeypatch.setattr("sit_lmpc.diagnostics.plots.LivePlots", Live)
    config = load_config(overrides={
        "simulation": {"seed_mode": "fixed", "seed": 7, "max_steps": 3},
        "planner": {"samples": 8, "horizon": 2, "hull_precompile": False,
                    "dynamics": {"backend": "st"}, "belief": {"enabled": False},
                    "risk_constraint": {"enabled": False}, "probabilistic_safety": {"enabled": False}},
        "risk": {"enabled": False},
        "diagnostics": {"trace_steps": 5 if trace else 0, "plots": live, "plot_interval": 2,
                        "render": False, "summary_plots": False, "keyboard_controls": False},
        "output": {"directory": str(tmp_path / "run"), "resume": None, "quick_load": False},
    })
    result = runner.run(config, tracking_only=True)
    assert result.status == "step_limit" and len(calls) == expected
    assert len(displayed) == (2 if live else 0)
    records = json.loads((result.directory / "trace.json").read_text())
    assert len(records) == (3 if trace else 0)
    if trace:
        for record in records:
            assert len(record["predicted"]) == 7 and np.isfinite(record["predicted"]).all()
