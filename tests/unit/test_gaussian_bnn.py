"""Small synthetic population fixtures, no dependence on local trained results."""

from dataclasses import replace
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dyna_dist.config import ModelConfig
from dyna_dist.data.transforms import Normalizer
from dyna_dist.inference.posterior import PopulationModel
from dyna_dist.io.checkpoint import save_model
from dyna_dist.models.features import initialize
from sit_lmpc.config import load_config
from sit_lmpc.config.schema import DynamicsConfig, ExperimentConfig
from sit_lmpc.dynamics.gaussian_bnn import BNNScenarioProvider, load_predictor
from sit_lmpc.dynamics.ensemble import member_rollouts, predictor_rollout, predictor_step, weighted_states


@pytest.fixture
def float32_runtime():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def bnn_checkpoint(tmp_path, float32_runtime):
    config = ModelConfig(mode="dynamics", latent_dimension=4, hidden_sizes=(8,),
                         dtype="float32", integrator_timestep=0.01)
    model = PopulationModel(
        initialize(config, jax.random.key(1)),
        Normalizer(jnp.zeros(6, dtype=jnp.float32), jnp.ones(6, dtype=jnp.float32),
                   jnp.asarray([1, 0, 0, 0, 0, 0, 0], dtype=jnp.float32),
                   jnp.full(7, 0.005, dtype=jnp.float32)),
        jnp.zeros(19, dtype=jnp.float32), config,
    )
    return save_model(model, tmp_path / "trained")


def load_fixture(path, count=4):
    return load_predictor(DynamicsConfig("gaussian_bnn", str(path), count),
                          replace(ExperimentConfig().simulation, dtype="float32"),
                          jax.random.key(2))


def test_population_predictor_coherence_and_physical_units(bnn_checkpoint):
    model = load_fixture(bnn_checkpoint)
    state = jnp.asarray([0, 0, 0.1, 10, 0.3, 0, 0], dtype=jnp.float32)
    actions = jnp.full((3, 2), 0.2, dtype=jnp.float32)
    members = member_rollouts(model, state, actions)
    expected = model.functions.rollout(state, actions * model.action_scale)[0]
    np.testing.assert_array_equal(members, expected)
    np.testing.assert_allclose(predictor_rollout(model, state, actions),
                               weighted_states(expected, model.weights), atol=1e-6)
    np.testing.assert_allclose(predictor_step(model, state, actions[0]),
                               weighted_states(expected[:, 0], model.weights), atol=1e-6)
    provider = BNNScenarioProvider(model, 32)
    controls = jnp.stack([actions, actions * 0.5, actions])
    paths = provider.rollout(state, controls, key=jax.random.key(3))
    np.testing.assert_array_equal(paths.states[0], paths.states[2])
    np.testing.assert_array_equal(paths.states[0], members)
    assert paths.probability_error_bound is None
    reordered = provider.rollout(state, controls[::-1], key=jax.random.key(3))
    np.testing.assert_array_equal(reordered.states, paths.states[::-1])
    np.testing.assert_array_equal(model.resample(jax.random.key(5)).params,
                                  model.resample(jax.random.key(5)).params)
    assert not np.array_equal(model.resample(jax.random.key(5)).params, model.params)


def test_temporary_transition_conditioner_uses_same_features_and_never_changes_belief(bnn_checkpoint):
    from dyna_dist.inference.hypothetical import condition_transitions, condition_factor_batch
    from dyna_dist.inference.posterior import components
    from dyna_dist.models.features import prior_moments_and_variance

    model = load_fixture(bnn_checkpoint)
    population = model.functions.model
    mean, factor, variance = prior_moments_and_variance(population.parameters, population.config)
    states = jnp.broadcast_to(jnp.asarray([0, 0, .1, 10, 0, 0, 0], jnp.float32), (2, 3, 7))
    controls = jnp.full((2, 3, 2), .1, dtype=jnp.float32)  # Physical controls, not normalized actions.
    targets = states.at[..., 0].add(1.)
    mask = jnp.asarray([[1, 1, 1], [1, 1, 0]], dtype=jnp.float32)
    base, basis = components(population, states, controls)
    residual = population.targets(states, controls, targets) - base
    expected = condition_factor_batch(mean, factor, basis, residual, variance, mask, chunk_size=2)
    original = np.asarray(model.params).copy()
    actual = condition_transitions(population, mean, factor,
        states.at[1, -1].set(jnp.nan), controls.at[1, -1].set(jnp.nan),
        targets.at[1, -1].set(jnp.nan), mask, chunk_size=2)
    assert np.asarray(actual[2]).all() and model.posterior_mean is None
    for x, y in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_allclose(x, y, atol=2e-5)
    np.testing.assert_array_equal(model.params, original)


def test_resampling_reuses_host_metadata_but_new_belief_invalidates_it(bnn_checkpoint):
    model = load_fixture(bnn_checkpoint)
    provider = BNNScenarioProvider(model, chance_estimation="planning_samples")
    metadata = provider.metadata
    resampled = provider.with_model(model.resample(jax.random.key(9)))
    assert resampled.metadata is metadata
    changed = resampled.with_model(resampled.model.replace(belief_generation=jnp.asarray(3)))
    assert changed.metadata is not metadata and changed.metadata["belief_generation"] == 3


def test_independent_verification_and_honest_bound(bnn_checkpoint):
    model = load_fixture(bnn_checkpoint)
    provider = BNNScenarioProvider(model, 32, 0.95)
    state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=jnp.float32)
    controls = jnp.zeros((3, 2, 2), dtype=jnp.float32)
    key = jax.random.key(5)
    with pytest.raises(ValueError, match="independent"):
        provider.verification_rollout(state, controls, planning_key=key, validation_key=key)
    paths = provider.verification_rollout(state, controls, planning_key=key,
                                         validation_key=jax.random.key(6))
    assert paths.states.shape == (3, 32, 2, 7)
    assert 0 < float(paths.probability_error_bound) < 1
    np.testing.assert_allclose(paths.weights, 1 / 32)


def test_planning_sample_provider_is_explicitly_empirical(bnn_checkpoint):
    provider = BNNScenarioProvider(load_fixture(bnn_checkpoint, 64), chance_estimation="planning_samples")
    assert not provider.uses_independent_verification
    assert provider.metadata["sample_count"] == 64
    assert provider.metadata["verification_sample_count"] == 0
    assert provider.metadata["confidence"] is None
    assert provider.metadata["verification_confidence"] is None
    assert provider.metadata["probability_estimator"] == "empirical_planning_samples"
    with pytest.raises(ValueError, match="disabled"):
        provider.verification_rollout(None, None, planning_key=None, validation_key=None)


def test_keyboard_comparison_reuses_current_planning_functions_without_sampling(
    bnn_checkpoint, monkeypatch
):
    from sit_lmpc.diagnostics.model_comparison import ModelComparisons
    from sit_lmpc.experiments.runner import build_vehicle_dynamics

    model = load_fixture(bnn_checkpoint).replace(belief_generation=jnp.asarray(7))
    config = replace(ExperimentConfig(), simulation=replace(
        ExperimentConfig().simulation, dtype="float32"))
    _, truth = build_vehicle_dynamics(config)
    comparison = ModelComparisons(model, truth, (0.3, 0.7), 3)
    assert comparison.belief_generation == 7
    assert comparison.weight_kind == "gaussian_function_samples"
    current = model.resample(jax.random.key(14), count=6).replace(belief_generation=jnp.asarray(8))
    coefficients = np.asarray(current.params).copy()
    comparison.belief = current  # The runner freezes its most recent planning batch here.
    monkeypatch.setattr(type(model), "resample", lambda *args, **kwargs: pytest.fail("plot resampled belief"))
    state = np.asarray([0, 0, 0.1, 10, 0.3, 0, 0], dtype=np.float32)
    actions = np.full((3, 2), 0.2, dtype=np.float32)
    first = comparison.planned(state, actions, 12)
    second = comparison.planned(state, actions, 12)
    assert first.candidates.shape == (6, 4, 7) and first.frictions == (None,) * 6
    assert first.belief_generation == 8 and first.belief_mode == "gaussian_bnn"
    np.testing.assert_allclose(first.candidates[:, 1:], member_rollouts(current, state, actions))
    np.testing.assert_allclose(first.belief[1:], predictor_rollout(current, state, actions))
    np.testing.assert_array_equal(first.candidates, second.candidates)
    np.testing.assert_array_equal(first.model_weights, current.weights)
    np.testing.assert_array_equal(current.params, coefficients)


def test_config_roundtrip_resume_overrides_and_interval(bnn_checkpoint, tmp_path):
    from sit_lmpc.config.schema import load_config as read
    import yaml

    config = load_config(overrides={
        "planner": {"sit_lmpc_mode": False, "dynamics": {"backend": "gaussian_bnn", "checkpoint": str(bnn_checkpoint)},
                    "belief": {"enabled": False}},
    })
    hash(config.planner)
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(config.as_dict()))
    restored = read(file, {"planner": {"dynamics": {"particles": 8, "chance_estimation": "planning_samples"}}})
    assert restored.planner.dynamics.particles == 8
    assert restored.planner.dynamics.checkpoint == str(bnn_checkpoint)
    assert restored.planner.dynamics.chance_estimation == "planning_samples"
    assert read(file).planner.dynamics.chance_estimation == "independent"
    from sit_lmpc.config.resume import resume_overrides
    assert resume_overrides({"planner": {"dynamics": {"chance_estimation": "planning_samples"}}}) == {
        "planner": {"dynamics": {"chance_estimation": "planning_samples"}}}
    with pytest.raises(ValueError, match="chance_estimation"):
        read(file, {"planner": {"dynamics": {"chance_estimation": "unknown"}}})
    with pytest.raises(ValueError, match="interval"):
        load_predictor(config.planner.dynamics, replace(config.simulation, timestep=0.2),
                       jax.random.key(1))
    with pytest.raises(ValueError, match="frozen"):
        replace(config, planner=replace(config.planner,
                 belief=replace(config.planner.belief, enabled=True))).validate()


def test_artifact_copy_survives_original_removal(bnn_checkpoint, tmp_path):
    config = DynamicsConfig("gaussian_bnn", str(bnn_checkpoint), 4)
    simulation = replace(ExperimentConfig().simulation, dtype="float32")
    copy = tmp_path / "portable"
    original = load_predictor(config, simulation, jax.random.key(2), destination=copy)
    shutil.rmtree(bnn_checkpoint)
    restored = load_predictor(config, simulation, jax.random.key(2), source=copy)
    assert restored.functions.model.fingerprint == original.functions.model.fingerprint
    np.testing.assert_array_equal(original.params, restored.params)


def test_runtime_float64_cast_retains_artifact(bnn_checkpoint):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        model = load_predictor(DynamicsConfig("gaussian_bnn", str(bnn_checkpoint), 4),
                               replace(ExperimentConfig().simulation, dtype="float64"),
                               jax.random.key(2))
        assert model.params.dtype == jnp.float64
        assert model.functions.model.config.dtype == "float64"
        assert model.functions.model.normalizer.input_mean.dtype == jnp.float64
        state = jnp.asarray([0, 0, 0, 10, 0, 0, 0], dtype=jnp.float64)
        assert np.isfinite(predictor_step(model, state, jnp.zeros(2))).all()
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_full_checkpoint_bundles_model_and_resumes_without_source(bnn_checkpoint, tmp_path):
    import yaml
    from test_experiment_checkpoint import empty_experiment
    from sit_lmpc.data.experiment_checkpoint import save_checkpoint, restore_checkpoint, load_checkpoint_config

    directory = tmp_path / "run"
    state = empty_experiment(directory)
    config = yaml.safe_load((directory / "config.yaml").read_text())
    config["planner"]["dynamics"] = {"backend": "gaussian_bnn", "checkpoint": str(bnn_checkpoint), "particles": 4}
    config["planner"]["belief"]["enabled"] = False
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    dynamics = DynamicsConfig("gaussian_bnn", str(bnn_checkpoint), 4)
    simulation = replace(ExperimentConfig().simulation, dtype="float32")
    original = load_predictor(dynamics, simulation, jax.random.key(2),
                              destination=directory / "dynamics_model")
    checkpoint = save_checkpoint(directory, 0, state, {"cartesian": np.zeros(7)}, False, False)
    shutil.rmtree(bnn_checkpoint)
    restored_config = load_checkpoint_config(checkpoint, {"planner": {"dynamics": {"checkpoint": None, "particles": 8}}})
    resumed = tmp_path / "resumed"
    restore_checkpoint(resumed, checkpoint, state)
    loaded = load_predictor(restored_config.planner.dynamics, simulation, jax.random.key(2),
                            source=resumed / "dynamics_model")
    assert len(loaded.weights) == 8
    assert loaded.functions.model.fingerprint == original.functions.model.fingerprint


def test_mppi_resamples_once_per_control_update_and_reuses_jit(bnn_checkpoint, monkeypatch):
    from sit_lmpc.planners import mppi
    from sit_lmpc.planners.mppi import MPPI, tracking_control
    from sit_lmpc.track import Track, map_path

    model = load_fixture(bnn_checkpoint)
    config = replace(ExperimentConfig().planner, horizon=3, samples=8, iterations=2)
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = jnp.asarray([pose[0], pose[1], 0, 10, pose[2], 0, 0], dtype=jnp.float32)
    controller = MPPI(config, model, track)
    observed = []

    def observed_control(model, *args):
        observed.append(np.asarray(model.params).copy())
        return tracking_control(model, *args)

    monkeypatch.setattr(mppi, "tracking_control", observed_control)
    first = controller.update(state, 10)
    coefficients = np.asarray(controller.model.params)
    cache_size = tracking_control._cache_size()
    second = controller.update(state, 10)
    assert tracking_control._cache_size() == cache_size
    assert not np.array_equal(coefficients, controller.model.params)
    assert len(observed) == 2  # Entire optimization-iteration scan shares this batch.
    assert not np.array_equal(observed[0], observed[1])
    assert np.isfinite(second.action).all()
    assert first.model_trajectories.shape == (4, 3, 7)


@pytest.mark.parametrize("chance,risk", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mode", ["all", "terminal"])
@pytest.mark.parametrize("estimation", ["independent", "planning_samples"])
def test_lmppi_bnn_toggles_and_verified_candidates(bnn_checkpoint, monkeypatch, chance, risk, mode, estimation):
    from sit_lmpc.planners import lmppi
    from sit_lmpc.track import Track, map_path

    model = load_fixture(bnn_checkpoint)
    base = ExperimentConfig()
    config = replace(base.planner, horizon=3, samples=8, lambda_count=2,
                     mean_value=not (risk and mode == "terminal"),
                     hull_precompile_limit=40, hull_selection_points=30,
                     probabilistic_safety=replace(base.planner.probabilistic_safety,
                         enabled=chance, boundary_threshold=1.0, terminal_hull_threshold=1.0,
                         max_candidates=3),
                     risk_constraint=replace(base.planner.risk_constraint, enabled=risk,
                         evaluation="distribution", mode=mode, max_candidates=3))
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = jnp.asarray([pose[0], pose[1], 0, 10, pose[2], 0, 0], dtype=jnp.float32)
    verification_count = 256 if chance and mode == "all" else 16
    resamples = []
    resample = type(model).resample

    def tracked_resample(model, key, count=None):
        resamples.append(len(model.weights) if count is None else count)
        return resample(model, key, count)

    monkeypatch.setattr(type(model), "resample", tracked_resample)
    monkeypatch.setattr(BNNScenarioProvider, "rollout",
                        lambda *a, **k: pytest.fail("already-computed planning paths were not reused"))
    if estimation == "planning_samples":
        monkeypatch.setattr(BNNScenarioProvider, "verification_rollout",
                            lambda *a, **k: pytest.fail("empirical mode ran independent verification"))
    config = replace(config, dynamics=replace(config.dynamics, chance_estimation=estimation))
    input_mode = "state_action" if risk else "state"
    controller = lmppi.LMPPI(config, model, track,
                             value_input_mode=input_mode,
                             scenario_provider=BNNScenarioProvider(model, verification_count,
                                                                   chance_estimation=estimation))
    monkeypatch.setattr(lmppi, "predict_value", lambda snapshot, states, *args: jnp.zeros(len(states)))
    monkeypatch.setattr(lmppi, "distribution_risk_probabilities",
                        lambda snapshot, scenarios, states, *args: jnp.zeros((states.shape[0], states.shape[2])))

    def select(terminals, speed):
        points = terminals[0, :5] + np.random.default_rng(17).normal(size=(30, 5)) * [4, 1, .2, 3, .2]
        return np.column_stack((points, np.zeros((30, 3)))), np.zeros((2, 7))

    snapshot = SimpleNamespace(input_mode=input_mode, generation=1)
    result = controller.update(state, np.asarray([0, 0, 0, 10, 0, 0, 0]), snapshot,
        SimpleNamespace(select=select), risk_snapshot=snapshot if risk else None,
        risk_cutoff={"status": "ready", "generation": 1, "threshold": 1.0} if risk else None)
    assert np.isfinite(result.action).all()
    assert len(result.lambdas) == 2 + 2 * chance + risk
    assert result.model_trajectories.shape == (4, 3, 7)
    assert controller.actions.shape == (3 + int(risk), 2)
    assert resamples == ([4, verification_count] if chance and estimation == "independent" else [4])
    if chance:
        assert result.metrics["probabilistic_safety_status"] == "active"
        if estimation == "independent":
            assert result.metrics["verification_sample_count"] == verification_count
            assert result.metrics["chance_probability_error_bound"] > 0
            assert result.metrics["boundary_probability_upper"] is not None
        else:
            assert result.metrics["verification_sample_count"] == 0
            assert result.metrics["sample_count"] == 4
            assert result.metrics["chance_probability_error_bound"] is None
            assert result.metrics["boundary_probability_upper"] is None
            assert result.metrics["terminal_hull_probability_upper"] is None
            assert result.metrics["confidence"] is None
            assert result.metrics["probability_estimator"] == "empirical_planning_samples"
