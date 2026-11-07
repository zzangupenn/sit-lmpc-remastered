"""Explicit bounded runner configs; portable posterior and old-mode continuation."""

from dataclasses import replace
import json
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dyna_dist.config import ModelConfig
from dyna_dist.data.transforms import Normalizer
from dyna_dist.inference.posterior import PopulationModel
from dyna_dist.io.checkpoint import save_model
from dyna_dist.models.features import initialize
from sit_lmpc.config.schema import DynamicsConfig
from sit_lmpc.data.trajectories import load_trajectories
from sit_lmpc.experiments import runner
from test_stage2 import setup_run


def setup_posterior_run(tmp_path, monkeypatch, mode="posterior"):
    config, captured = setup_run(tmp_path, monkeypatch, failures=(3,), max_steps=27, max_laps=3)
    model_config = ModelConfig(mode="dynamics", latent_dimension=4, hidden_sizes=(8,), dtype="float32")
    model = PopulationModel(
        initialize(model_config, jax.random.key(17)),
        Normalizer(jnp.zeros(6, jnp.float32), jnp.ones(6, jnp.float32),
                   jnp.asarray([1, 0, 0, 0, 0, 0, 0], jnp.float32), jnp.full(7, .005, jnp.float32)),
        jnp.zeros(19, jnp.float32), model_config)
    checkpoint = save_model(model, tmp_path / "trained")
    config = replace(config,
        simulation=replace(config.simulation, dtype="float32", state_noise=0, control_noise=(0., 0.)),
        planner=replace(config.planner, dynamics=DynamicsConfig(
            backend="gaussian_bnn", checkpoint=str(checkpoint), particles=4,
            belief_update=mode, posterior_method="standard", update_batch_size=8,
            validation_particles=8, validation_max_windows=4)),
        output=replace(config.output, checkpoint_every_trials=1))
    original_mppi, original_lmppi = runner.MPPI, runner.LMPPI

    class Tracking(original_mppi):
        def __init__(self, planner, predictor, *args, **kwargs):
            super().__init__(planner, predictor, *args, **kwargs)
            self.model = predictor

    class Learning(original_lmppi):
        def __init__(self, planner, predictor, *args, **kwargs):
            super().__init__(planner, predictor, *args, **kwargs)
            self.model = predictor

    monkeypatch.setattr(runner, "MPPI", Tracking)
    monkeypatch.setattr(runner, "LMPPI", Learning)
    return config, captured


def test_attempt_boundary_generation_and_independent_posterior_resume(tmp_path, monkeypatch):
    config, _ = setup_posterior_run(tmp_path, monkeypatch)
    result = runner.run(config, tracking_only=True)
    assert result.status == "lap_limit" and result.trials == 3 and result.failures == 1
    failure = load_trajectories(result.directory, "failure")[0]
    success = load_trajectories(result.directory, "success")
    assert failure.belief_generations.tolist() == [0] * 3
    assert success[0].belief_generations.tolist() == [1] * 12
    assert success[1].belief_generations.tolist() == [2] * 12
    checkpoint = result.directory / "checkpoints/trial-000003"
    state = json.loads((checkpoint / "dynamics_posterior.json").read_text())
    assert state["generation"] == 3 and state["training_transitions"] == 20
    assert (checkpoint / "dynamics_model/model.msgpack").is_file()
    assert not (checkpoint / "dynamics_belief.json").exists()
    portable = tmp_path / "portable-checkpoint"
    shutil.copytree(checkpoint, portable)
    result.directory.rename(tmp_path / "archived-run")
    (tmp_path / "trained").rename(tmp_path / "archived-trained-model")
    resumed = replace(config,
        simulation=replace(config.simulation, max_steps=39, max_laps=4),
        planner=replace(config.planner, dynamics=replace(config.planner.dynamics, checkpoint=None)),
        output=replace(config.output, resume=str(portable), directory=str(tmp_path / "resumed")))
    continued = runner.run(resumed, tracking_only=True)
    assert continued.trials == 4 and continued.status == "lap_limit"
    last = load_trajectories(continued.directory, "success")[-1]
    assert last.belief_generations.tolist() == [3] * 12
    state = json.loads((continued.directory / "dynamics_posterior.json").read_text())
    assert state["generation"] == 4 and state["training_transitions"] == 29
    assert state["processed"].keys() == {"1", "2", "3", "4"}


def test_calibrated_runner_portable_resume_retains_evidence_and_partition(tmp_path, monkeypatch):
    from sit_lmpc.data.dynamics import trajectory_transitions
    from sit_lmpc.training.bnn_calibration import split_calibrated

    config, _ = setup_posterior_run(tmp_path, monkeypatch)
    config = replace(config, planner=replace(config.planner, dynamics=replace(
        config.planner.dynamics, posterior_method="noise_tempered", calibration_horizon=3,
        calibration_min_windows=1, calibration_powers=(1., .01))))
    result = runner.run(config, tracking_only=True)
    checkpoint = result.directory / "checkpoints/trial-000003"
    state = json.loads((checkpoint / "dynamics_posterior.json").read_text())
    records = [t for outcome in ("success", "failure")
               for t in load_trajectories(checkpoint, outcome)]
    count = sum(len(split_calibrated(trajectory_transitions(t, .02),
                    config.planner.dynamics)[0]) for t in records)
    assert state["schema"] == 2 and state["training_transitions"] == count
    with np.load(checkpoint / "dynamics_posterior.npz") as arrays:
        assert arrays["gram_components"].shape == (7, 4, 4)
        assert set(map(tuple, arrays["calibration_ids"])) & set(map(tuple, arrays["validation_ids"])) == set()
    portable = tmp_path / "portable-calibrated"
    shutil.copytree(checkpoint, portable)
    result.directory.rename(tmp_path / "old-calibrated-run")
    (tmp_path / "trained").rename(tmp_path / "old-population-model")
    resumed = replace(config,
        simulation=replace(config.simulation, max_steps=39, max_laps=4),
        planner=replace(config.planner, dynamics=replace(config.planner.dynamics, checkpoint=None)),
        output=replace(config.output, resume=str(portable), directory=str(tmp_path / "calibrated-resumed")))
    continued = runner.run(resumed, tracking_only=True)
    final = json.loads((continued.directory / "dynamics_posterior.json").read_text())
    assert continued.trials == 4 and continued.status == "lap_limit"
    last = load_trajectories(continued.directory, "success")[-1]
    added = len(split_calibrated(trajectory_transitions(last, .02), config.planner.dynamics)[0])
    assert final["training_transitions"] == count + added


def test_old_frozen_checkpoint_can_enable_posterior_learning(tmp_path, monkeypatch):
    config, _ = setup_posterior_run(tmp_path, monkeypatch, mode="frozen")
    result = runner.run(config, tracking_only=True)
    checkpoint = result.directory / "checkpoints/trial-000003"
    assert not (checkpoint / "dynamics_posterior.json").exists()
    resumed = replace(config,
        simulation=replace(config.simulation, max_steps=39, max_laps=4),
        planner=replace(config.planner, dynamics=replace(config.planner.dynamics, belief_update="posterior")),
        output=replace(config.output, resume=str(checkpoint), directory=str(tmp_path / "adapted")))
    continued = runner.run(resumed, tracking_only=True)
    state = json.loads((continued.directory / "dynamics_posterior.json").read_text())
    assert continued.trials == 4 and state["training_transitions"] == 29
    assert load_trajectories(continued.directory, "success")[-1].belief_generations.tolist() == [1] * 12
