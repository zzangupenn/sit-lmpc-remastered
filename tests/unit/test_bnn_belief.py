from dataclasses import replace
import json
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dyna_dist.inference.online import GaussianEvidence
from sit_lmpc.config import load_config
from sit_lmpc.config.schema import DynamicsConfig, ExperimentConfig
from sit_lmpc.data.dynamics import initial_transitions, trajectory_transitions
from sit_lmpc.data.initialization import InitialSnapshot, load_initial, save_initial
from sit_lmpc.data.trajectories import TrajectoryDataset
from sit_lmpc.dynamics.gaussian_bnn import BNNScenarioProvider
from sit_lmpc.experiments.randomness import ExperimentRandomness
from sit_lmpc.training.gaussian_bnn import GaussianBNNLearner
from test_gaussian_bnn import bnn_checkpoint as checkpoint_fixture, float32_runtime as runtime_fixture, load_fixture
from test_dynamics_belief import trajectory

bnn_checkpoint = checkpoint_fixture
float32_runtime = runtime_fixture


def learner(path, checkpoint, **settings):
    path.mkdir(exist_ok=True)
    model = load_fixture(checkpoint)
    # These tests exercise the explicitly selected legacy analytic update.
    settings = {"posterior_method": "standard", **settings}
    config = replace(DynamicsConfig(), backend="gaussian_bnn", belief_update="posterior",
                     update_batch_size=8, validation_particles=8, validation_max_windows=4,
                     **settings)
    return GaussianBNNLearner(config, model, replace(ExperimentConfig().simulation, dtype="float32"),
                              path, ExperimentRandomness(18).fork("dynamics_bnn_posterior"))


def test_exact_new_evidence_once_and_no_failure_reset(bnn_checkpoint, tmp_path):
    fit = learner(tmp_path, bnn_checkpoint)
    first = trajectory()
    fit.observe(first)
    assert fit.update() and fit.generation == 1
    assert fit.evidence.count == 9
    expected = GaussianEvidence.create(fit.evidence.mean, fit.evidence.factor, fit.evidence.variance)
    training, held_out = fit._split(trajectory_transitions(first, .02))
    for basis, residual in fit._features(training):
        expected.add(basis, residual)
    np.testing.assert_allclose(fit.snapshot.mean, expected.moments()[0])
    np.testing.assert_allclose(fit.snapshot.factor, expected.moments()[1])
    assert len(held_out) == 3
    assert fit.history[-1]["validation_after"]["transitions"] == 3
    assert set(fit.history[-1]["validation_after"]["horizons"]) == {"1"}
    fit.observe(first)
    assert not fit.update() and fit.evidence.count == 9 and fit.generation == 1
    second = trajectory(2, "failure")
    fit.observe(second)
    assert fit.update() and fit.evidence.count == 18 and fit.generation == 2
    training, _ = fit._split(trajectory_transitions(second, .02))
    for basis, residual in fit._features(training):
        expected.add(basis, residual)
    np.testing.assert_allclose(fit.snapshot.mean, expected.moments()[0])
    corrupted = replace(first, controls=first.controls + .01)
    with pytest.raises(ValueError, match="changed"):
        fit.observe(corrupted)


def test_posterior_sampling_and_verification_use_same_updated_distribution(bnn_checkpoint, tmp_path):
    fit = learner(tmp_path, bnn_checkpoint)
    fit.observe(trajectory())
    fit.update()
    model = fit.predictor()
    key = jax.random.key(21)
    sampled = model.resample(key, 16)
    expected = model.posterior_mean + jnp.matmul(
        jax.random.normal(key, (16, 4), dtype=model.params.dtype), model.posterior_factor.T,
        precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_array_equal(sampled.params, expected)
    assert model.context()["belief_generation"] == 1
    assert model.context()["adaptation"] == "gaussian_posterior"
    provider = BNNScenarioProvider(model, 16)
    state = jnp.asarray([0, 0, .1, 12, .2, 0, 0], dtype=model.params.dtype)
    controls = jnp.zeros((2, 3, 2), dtype=model.params.dtype)
    paths = provider.verification_rollout(state, controls, planning_key=jax.random.key(2), validation_key=key)
    direct = sampled.functions.rollout(state, controls * sampled.action_scale)
    np.testing.assert_array_equal(paths.states, direct)
    assert provider.metadata["probability_source"] == "gaussian_latent_posterior"
    assert fit.template.posterior_mean is None


def test_portable_restore_power_change_and_validation_rebuild(bnn_checkpoint, tmp_path):
    source = tmp_path / "source"
    fit = learner(source, bnn_checkpoint)
    dataset = TrajectoryDataset(source, ExperimentConfig(), 12)
    for record in (trajectory(), trajectory(2, "failure")):
        dataset.add(record)
        fit.observe(record)
        fit.update()
    portable = tmp_path / "checkpoint"
    shutil.copytree(source, portable)
    source.rename(tmp_path / "archived-source")
    resumed = learner(tmp_path / "resumed", bnn_checkpoint)
    resumed.restore(portable, .02)
    assert resumed.generation == 2 and resumed.evidence.count == 18
    np.testing.assert_array_equal(resumed.evidence.gram, fit.evidence.gram)
    np.testing.assert_array_equal(resumed.snapshot.mean, fit.snapshot.mean)
    assert len(resumed.history) == 2
    tempered = learner(tmp_path / "tempered", bnn_checkpoint, likelihood_power=.1)
    tempered.restore(portable, .02)
    np.testing.assert_allclose(tempered.snapshot.factor, fit.evidence.moments(.1)[1])
    assert tempered.evidence.count == 18 and tempered.generation == 3
    assert tempered.predictor().context()["inference"] == "generalized_bayes"
    changed = learner(tmp_path / "changed", bnn_checkpoint, validation_fraction=.4)
    changed.restore(portable, .02)
    assert changed.evidence.count == 14 and changed.generation == 3


def test_old_extension_gap_is_not_a_physical_transition():
    states = np.zeros((5, 8))
    states[:, 0], states[:, 3], states[:, -1] = [1, 2, 3, 90, 91], 10, [.1, .2, .3, 9., 9.1]
    initial = InitialSnapshot((states,), (states,), states[-1, :7], np.zeros((2, 2)), {},
                              (np.zeros((5, 2)),), (np.ones(5, dtype=bool),))
    batch = initial_transitions(initial, .1, .02)
    assert len(batch) == 3
    np.testing.assert_allclose(batch.next_states[:, 0] - batch.states[:, 0], 1)
    assert batch.ids[:, 1].tolist() == [0, 1, 3]


def test_schema3_quick_load_keeps_original_physical_records(bnn_checkpoint, tmp_path):
    record = trajectory()
    states = np.column_stack((record.cartesian[1:], record.times[1:]))
    safe = type("Safe", (), {"frenet": [states], "cartesian": [states],
        "controls": [np.vstack((record.controls[1:], np.zeros((1, 2))))],
        "control_masks": [np.arange(12) < 11]})()
    config = replace(ExperimentConfig(), planner=replace(ExperimentConfig().planner, safe_set_size=1))
    path = tmp_path / "init.npz"
    save_initial(path, safe, record.cartesian[-1], np.zeros((10, 2)), config, trajectories=[record])
    initial = load_initial(path, config)
    assert initial.metadata["schema"] == 3
    batch = initial_transitions(initial, .1, .02)
    np.testing.assert_array_equal(batch.states, record.cartesian[:-1])
    np.testing.assert_array_equal(batch.controls, record.controls)
    fit = learner(tmp_path / "learned", bnn_checkpoint)
    fit.import_initial(initial)
    assert fit.evidence.count == 9
    assert (fit.initial.ids[:, 0] < 0).all()
    fit.import_initial(initial)
    assert fit.evidence.count == 9


@pytest.mark.parametrize("settings", [
    {"belief_update": "unknown"}, {"likelihood_power": 0}, {"likelihood_power": 1.1},
    {"likelihood_power": True}, {"validation_fraction": .5}, {"update_batch_size": 0},
])
def test_invalid_posterior_config(settings):
    with pytest.raises(ValueError):
        load_config(overrides={"planner": {"dynamics": {"backend": "gaussian_bnn", **settings},
                                          "belief": {"enabled": False}}})


def test_bnn_chance_preset_preserves_st_safety_semantics_without_bank_or_risk_learning():
    config = load_config("tests/fixtures/presets/bnn96-chance.yaml")
    baseline = load_config("tests/fixtures/presets/st-model-bank.yaml")
    dynamics = config.planner.dynamics
    assert dynamics.backend == "gaussian_bnn" and dynamics.belief_update == "posterior"
    assert "latent-96/best_refit" in dynamics.checkpoint
    assert dynamics.particles == 64 and dynamics.verification_particles == 2048
    assert dynamics.chance_estimation == "planning_samples"
    assert dynamics.verification_confidence == .95 and dynamics.likelihood_power == 1.
    assert dynamics.validation_fraction == .2
    assert not config.planner.belief.enabled and not config.planner.sit_lmpc_mode
    assert not config.risk.enabled and not config.planner.risk_constraint.enabled
    assert config.planner.probabilistic_safety.enabled
    assert config.planner.probabilistic_safety.fallback == "baseline"
    # User-editable particle/lambda budgets and experiment lengths need not be
    # identical between presets; their safety event definitions stay compatible.
    for field in ("enabled", "boundary_threshold", "terminal_hull_threshold", "fallback"):
        assert getattr(config.planner.probabilistic_safety, field) == getattr(baseline.planner.probabilistic_safety, field)
    for field in ("map", "friction", "timestep", "integrator_timestep", "dtype",
                  "initial_speed", "action_scale", "control_noise", "state_noise",
                  "max_steps", "seed_mode"):
        assert getattr(config.simulation, field) == getattr(baseline.simulation, field)
    for field in ("horizon", "safe_set_size", "half_width", "boundary_buffer",
                  "hull_precompile", "hull_precompile_limit", "safe_set_relaxation"):
        assert getattr(config.planner, field) == getattr(baseline.planner, field)
    assert config.value == baseline.value
    assert config.output.exp_name == "bnn96-posterior-chance-f05"
    assert not config.output.quick_load and config.output.resume is None
    assert config.diagnostics.render and not config.diagnostics.plots
    assert config.diagnostics.keyboard_controls
    assert config.diagnostics.summary_plots
    hash(config.planner)


def test_legacy_frozen_and_bank_config_stay_compatible(tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text("planner:\n  dynamics:\n    backend: gaussian_bnn\n  belief:\n    enabled: false\n")
    assert load_config(old).planner.dynamics.belief_update == "frozen"
    assert load_config(old).planner.dynamics.chance_estimation == "independent"
    config = load_config()
    assert config.planner.dynamics.backend == "st"
    assert config.planner.dynamics.belief_update == "frozen"
    assert not config.planner.belief.enabled


def test_nonfinite_update_keeps_previous_published_belief(bnn_checkpoint, tmp_path, monkeypatch):
    fit = learner(tmp_path, bnn_checkpoint)
    fit.observe(trajectory())
    old_mean, old_factor = fit.snapshot.mean.copy(), fit.snapshot.factor.copy()
    monkeypatch.setattr(fit, "_evaluate", lambda *args: (_ for _ in ()).throw(FloatingPointError("bad likelihood")))
    assert not fit.update()
    assert fit.evidence.count == 0 and fit.generation == 0
    np.testing.assert_array_equal(fit.snapshot.mean, old_mean)
    np.testing.assert_array_equal(fit.snapshot.factor, old_factor)
    assert fit.history[-1]["status"] == "rejected_nonfinite_update"
    assert json.loads((tmp_path / "dynamics_posterior.json").read_text())["generation"] == 0


def test_posterior_publications_reuse_mppi_jit(bnn_checkpoint, tmp_path):
    from sit_lmpc.planners.mppi import MPPI, tracking_control
    from sit_lmpc.track import Track, map_path

    fit = learner(tmp_path, bnn_checkpoint)
    config = replace(ExperimentConfig().planner, dynamics=fit.config, samples=8, horizon=3)
    track = Track.from_track_path(map_path("custom1"))
    pose = np.asarray(track.frenet_to_cartesian(0, 0, 0))
    state = jnp.asarray([pose[0], pose[1], .1, 12, pose[2], 0, 0], jnp.float32)
    planner = MPPI(config, fit.predictor(), track)
    tracking_control.clear_cache()
    for generation in range(3):
        planner.model = fit.predictor()
        assert np.isfinite(planner.update(state, 10).action).all()
        fit.observe(trajectory(generation + 1))
        fit.update()
    assert tracking_control._cache_size() == 1


def test_posterior_publications_retain_lmppi_hull_executables(bnn_checkpoint, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sit_lmpc.planners import lmppi
    from sit_lmpc.track import Track, map_path

    fit = learner(tmp_path, bnn_checkpoint)
    config = replace(ExperimentConfig().planner, dynamics=fit.config, samples=8, horizon=3,
                     lambda_count=2, hull_precompile_limit=40)
    track = Track.from_track_path(map_path("custom1"))
    pose = np.asarray(track.frenet_to_cartesian(0, 0, 0))
    state = jnp.asarray([pose[0], pose[1], .1, 12, pose[2], 0, 0], jnp.float32)
    controller = lmppi.LMPPI(config, fit.predictor(), track)
    controller.precompile_hulls()
    old = dict(controller.hull_evaluator.compiled)
    monkeypatch.setattr(lmppi, "predict_value", lambda snapshot, states, *args: jnp.zeros(len(states)))

    def select(terminals, speed):
        points = terminals[0, :5] + np.random.default_rng(17).normal(size=(35, 5)) * [2, .5, .1, 1, .1]
        return np.column_stack((points, np.zeros((35, 3)))), np.zeros((2, 7))

    lmppi.sample_rollouts.clear_cache()
    for generation in range(2):
        controller.model = fit.predictor()
        result = controller.update(state, np.asarray([0, 0, .1, 12, 0, 0, 0], np.float32),
            SimpleNamespace(input_mode="state", generation=1), SimpleNamespace(select=select))
        assert np.isfinite(result.action).all()
        fit.observe(trajectory(generation + 1))
        fit.update()
    assert lmppi.sample_rollouts._cache_size() == 1
    assert controller.hull_evaluator.compiled == old
