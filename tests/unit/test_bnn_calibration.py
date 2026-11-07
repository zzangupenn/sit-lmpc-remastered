from dataclasses import replace
import json
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dyna_dist.inference.posterior import FunctionSamples
from sit_lmpc.config import load_config
from sit_lmpc.config.schema import DynamicsConfig, ExperimentConfig
from sit_lmpc.data.dynamics import TransitionBatch, trajectory_transitions
from sit_lmpc.data.trajectories import Trajectory, TrajectoryDataset
from sit_lmpc.experiments.randomness import ExperimentRandomness
from sit_lmpc.training.bnn_calibration import split_calibrated, calibration_windows
from sit_lmpc.training.gaussian_bnn import GaussianBNNLearner
from test_gaussian_bnn import (
    bnn_checkpoint as checkpoint_fixture, float32_runtime as runtime_fixture, load_fixture,
)

bnn_checkpoint = checkpoint_fixture
float32_runtime = runtime_fixture


def settings(**changes):
    return replace(DynamicsConfig(), backend="gaussian_bnn", belief_update="posterior",
                   calibration_split="contiguous",
                   posterior_method="noise_tempered", calibration_horizon=3,
                   calibration_min_windows=2, update_batch_size=32,
                   validation_particles=8, validation_max_windows=8,
                   calibration_powers=(1., .01, .0001), **changes)


def make_learner(path, artifact, config=None):
    path.mkdir(exist_ok=True)
    return GaussianBNNLearner(config or settings(), load_fixture(artifact),
        replace(ExperimentConfig().simulation, dtype="float32"), path,
        ExperimentRandomness(18).fork("dynamics_bnn_posterior"))


def record(model, n=120, attempt=1):
    physical_controls = jnp.zeros((n, 2), jnp.float32)
    initial = jnp.asarray([0, 0, .1, 10, .2, 0, 0], jnp.float32)
    population = model.functions.model
    # Exactly one representable, unchanged function, without observation noise.
    functions = FunctionSamples(population, jnp.zeros((1, 4), jnp.float32))
    states = np.vstack((np.asarray(initial), np.asarray(functions.rollout(initial, physical_controls))[0, 0]))
    goal = np.zeros(n + 1, bool)
    goal[-1] = True
    return Trajectory(attempt, attempt, "success", "lap_completed", states, states.copy(),
        np.asarray(physical_controls), np.arange(n + 1) * .1, np.arange(n + 1),
        np.zeros(n + 1, bool), goal, np.full(n, "mppi"), np.zeros(n, int), np.arange(n),
        contacts=np.zeros(n, bool), integrator_timestep=.02)


def test_blocked_partition_and_nonoverlapping_windows(bnn_checkpoint):
    cfg = replace(settings(), calibration_horizon=10)
    batch = trajectory_transitions(record(load_fixture(bnn_checkpoint), n=400), .02)
    fitting, calibration, test, gaps = split_calibrated(batch, cfg)
    assert [len(x) for x in (fitting, calibration, test, gaps)] == [220, 80, 80, 20]
    joined = np.concatenate([x.ids for x in (fitting, calibration, test, gaps)])
    assert len(np.unique(joined, axis=0)) == len(batch)
    windows = calibration_windows(calibration, 10, 64)
    assert len(windows) == 8 and windows.mask.all()
    np.testing.assert_array_equal(windows.states, calibration.states[::10])
    missing = calibration.take(np.delete(np.arange(80), 5))
    windows = calibration_windows(missing, 10, 64)
    assert len(windows) == 7
    for window in windows.controls:
        assert window.shape == (10, 2)


def test_calibrated_update_does_not_train_or_select_on_test_targets(bnn_checkpoint, tmp_path):
    first = make_learner(tmp_path / "a", bnn_checkpoint)
    second = make_learner(tmp_path / "b", bnn_checkpoint)
    batch = trajectory_transitions(record(first.template), .02)
    _, _, test, _ = split_calibrated(batch, first.config)
    targets = batch.next_states.copy()
    targets[test.ids[:, 1]] += 100
    modified = replace(batch, next_states=targets)
    for learner, data in ((first, batch), (second, modified)):
        learner._stage(1, data)
        assert learner.update()
    np.testing.assert_array_equal(first.snapshot.mean, second.snapshot.mean)
    np.testing.assert_array_equal(first.snapshot.factor, second.snapshot.factor)
    assert first.active_power == second.active_power
    assert first.evidence.count == 66
    assert first.generation == 1
    assert len(first.calibration) == 24 and len(first.validation) == 24
    assert first.predictor().context()["inference"] == "calibrated_working_likelihood"


def test_held_update_retains_unique_evidence_for_next_attempt(bnn_checkpoint, tmp_path):
    learner = make_learner(tmp_path, bnn_checkpoint)
    short = trajectory_transitions(record(learner.template, n=12), .02)
    learner._stage(1, short)
    assert not learner.update() and learner.generation == 0
    count = learner.evidence.count
    assert count > 0
    assert learner.history[-1]["status"] == "held_insufficient_calibration"
    learner._stage(1, short)
    assert not learner.update() and learner.evidence.count == count
    learner._stage(2, trajectory_transitions(record(learner.template, attempt=2), .02))
    assert learner.update()
    assert learner.evidence.count == count + 66 and learner.generation == 1


def test_calibrated_checkpoint_and_mode_changes_rebuild_without_duplicates(bnn_checkpoint, tmp_path):
    source = tmp_path / "source"
    fit = make_learner(source, bnn_checkpoint)
    archive = TrajectoryDataset(source, ExperimentConfig(), 100)
    trajectory = record(fit.template)
    archive.add(trajectory)
    fit.observe(trajectory)
    assert fit.update()
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(source, checkpoint)
    source.rename(tmp_path / "old-source")
    restored = make_learner(tmp_path / "restored", bnn_checkpoint)
    restored.restore(checkpoint, .02)
    assert restored.evidence.count == 66 and restored.generation == fit.generation
    np.testing.assert_array_equal(restored.snapshot.mean, fit.snapshot.mean)
    np.testing.assert_array_equal(restored.snapshot.factor, fit.snapshot.factor)
    assert restored.active_power == fit.active_power
    changed = make_learner(tmp_path / "changed", bnn_checkpoint,
                           replace(settings(), likelihood_power=.01))
    changed.restore(checkpoint, .02)
    assert changed.evidence.count == 66
    if changed.history[-1]["published"]:
        assert changed.active_power <= .01
    else:
        # A changed policy can be rejected by the publication safeguard. The
        # previous explicitly logged belief is retained, not silently relabeled.
        assert changed.history[-1]["status"].startswith("held_")
        np.testing.assert_array_equal(changed.snapshot.mean, fit.snapshot.mean)
    legacy = make_learner(tmp_path / "legacy", bnn_checkpoint,
                          replace(settings(), posterior_method="standard"))
    legacy.restore(checkpoint, .02)
    assert legacy.evidence.count == 96
    assert json.loads((legacy.directory / "dynamics_posterior.json").read_text())["schema"] == 1
    TrajectoryDataset(legacy.directory, ExperimentConfig(), 100).add(trajectory)
    rebuilt = make_learner(tmp_path / "rebuilt", bnn_checkpoint)
    rebuilt.restore(legacy.directory, .02)
    assert rebuilt.evidence.count == 66


@pytest.mark.parametrize("fields", [
    {"posterior_method": "invalid"}, {"calibration_horizon": 0},
    {"calibration_fraction": 0}, {"calibration_target_coverage": 1.1},
    {"calibration_powers": [0]}, {"calibration_powers": [True]},
    {"calibration_score_slack": -1}, {"calibration_min_windows": 1000},
])
def test_invalid_calibration_configuration(fields):
    with pytest.raises(ValueError):
        load_config("tests/fixtures/presets/bnn96-calibrated.yaml",
                    {"planner": {"dynamics": fields}})


@pytest.mark.parametrize("preset", ["bnn-posterior", "bnn-chance"])
def test_bnn_posterior_presets_default_to_tempering(preset):
    config = load_config(f"configs/{preset}.yaml")
    assert config.planner.dynamics.posterior_method == "tempered"
    assert config.planner.dynamics.belief_update == "posterior"
    hash(config.planner)


def test_tempering_default_preserves_st_frozen_and_explicit_legacy_modes(tmp_path):
    assert DynamicsConfig().posterior_method == "tempered"
    ExperimentConfig().validate()
    default = load_config()
    assert default.planner.dynamics.backend == "st"
    assert default.planner.dynamics.belief_update == "frozen"
    assert default.planner.dynamics.posterior_method == "tempered"
    assert load_config("tests/fixtures/presets/bnn.yaml").planner.dynamics.belief_update == "frozen"
    partial = tmp_path / "partial.yaml"
    partial.write_text("planner:\n  dynamics:\n    backend: gaussian_bnn\n    belief_update: posterior\n  belief:\n    enabled: false\n")
    assert load_config(partial).planner.dynamics.posterior_method == "tempered"
    legacy = load_config(partial, {"planner": {"dynamics": {"posterior_method": "standard"}}})
    assert legacy.planner.dynamics.posterior_method == "standard"


def test_default_posterior_calibrates_power_without_changing_noise(bnn_checkpoint, tmp_path):
    config = replace(DynamicsConfig(), backend="gaussian_bnn", belief_update="posterior",
                     calibration_split="contiguous",
                     calibration_horizon=3, calibration_min_windows=2, update_batch_size=32,
                     validation_particles=8, validation_max_windows=8,
                     calibration_powers=(1., .01, .0001))
    learner = make_learner(tmp_path, bnn_checkpoint, config)
    assert learner.calibrated and learner.config.posterior_method == "tempered"
    learner.observe(record(learner.template))
    assert learner.update()
    assert learner.evidence.count == 66
    assert learner.history[-1]["posterior_method"] == "tempered"
    assert learner.active_power in config.calibration_powers
    np.testing.assert_array_equal(learner.active_variance, learner.evidence.variance)


def test_random_window_posterior_uses_seeded_partitions_and_rebuilds_changed_split(bnn_checkpoint, tmp_path):
    config = replace(settings(), calibration_split="random_windows")
    learner = make_learner(tmp_path/"source", bnn_checkpoint, config)
    trajectory = record(learner.template, n=180)
    archive = TrajectoryDataset(learner.directory, ExperimentConfig(), 100)
    archive.add(trajectory)
    batch = trajectory_transitions(trajectory, .02)
    fitting, calibration, test, _ = split_calibrated(batch, config, learner.randomness.seed)
    learner.observe(trajectory)
    learner.update()
    assert learner.evidence.count == len(fitting)
    np.testing.assert_array_equal(learner.calibration.ids, calibration.ids)
    assert not set(map(tuple, fitting.ids)) & set(map(tuple, calibration.ids))
    assert len(calibration_windows(calibration, 3, 64)) >= 3
    same = make_learner(tmp_path/"same", bnn_checkpoint, config)
    same.restore(learner.directory, .02)
    assert same.evidence.count == learner.evidence.count
    np.testing.assert_array_equal(same.snapshot.factor, learner.snapshot.factor)
    changed_config = replace(config, calibration_split="contiguous")
    changed = make_learner(tmp_path/"changed", bnn_checkpoint, changed_config)
    changed.restore(learner.directory, .02)
    assert changed.evidence.count == len(split_calibrated(batch, changed_config, 18)[0])
    assert changed.evidence.count != learner.evidence.count


def test_calibrated_publications_reuse_mppi_jit(bnn_checkpoint, tmp_path):
    from sit_lmpc.planners.mppi import MPPI, tracking_control
    from sit_lmpc.track import Track, map_path

    fit = make_learner(tmp_path, bnn_checkpoint)
    planner_config = replace(ExperimentConfig().planner, dynamics=fit.config, samples=8, horizon=3)
    track = Track.from_track_path(map_path("custom1"))
    pose = np.asarray(track.frenet_to_cartesian(0, 0, 0))
    state = jnp.asarray([pose[0], pose[1], .1, 10, pose[2], 0, 0], jnp.float32)
    controller = MPPI(planner_config, fit.predictor(), track)
    tracking_control.clear_cache()
    for attempt in range(1, 4):
        controller.model = fit.predictor()
        assert np.isfinite(controller.update(state, 10).action).all()
        fit.observe(record(fit.template, attempt=attempt))
        fit.update()
    assert fit.generation >= 1
    assert tracking_control._cache_size() == 1
