"""Real track/ensemble gates, bounded GPU shapes, portable resume and clean stops."""

from dataclasses import replace
import json
import shutil
from types import SimpleNamespace
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import (
    BeliefConfig,
    ExperimentConfig,
    ModelBankConfig,
    ProbabilisticSafetyConfig,
    RiskConstraintConfig,
)
from sit_lmpc.data.normalization import Normalization
from sit_lmpc.data.trajectories import load_trajectories
from sit_lmpc.dynamics.bank import STModelBank
from sit_lmpc.dynamics.belief import BeliefSnapshot
from sit_lmpc.dynamics.distribution import STBankScenarioProvider
from sit_lmpc.dynamics.ensemble import STEnsemble
from sit_lmpc.experiments import runner
from sit_lmpc.planners import lmppi
from sit_lmpc.planners.probabilistic import SafetyStop, chance_probabilities
from sit_lmpc.track import Track, map_path
from sit_lmpc.training.risk import RiskSnapshot, initial_state
from sit_lmpc.training.value import ValueTrainer
from test_stage2 import setup_run


@pytest.fixture
def float32_runtime():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "chance,risk", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("mode", ["all", "terminal"])
@pytest.mark.parametrize("hull_query", ["facets", "implicit"])
def test_real_track_branches_and_network_reuse_bounded_kernels(
    monkeypatch, chance, risk, mode, float32_runtime, hull_query
):
    base = ExperimentConfig()
    base = replace(
        base, simulation=replace(base.simulation, dtype="float32", seed=11, seed_mode="fixed")
    )
    config = replace(
        base.planner,
        horizon=3,
        samples=16,
        lambda_count=2,
        hull_precompile=True,
        hull_precompile_limit=40,
        safe_set_query=hull_query,
        hull_selection_points=64,
        probabilistic_safety=ProbabilisticSafetyConfig(
            enabled=chance,
            boundary_threshold=0.5,
            terminal_hull_threshold=1.0,
            lambda_count=2,
            max_candidates=3,
            fallback="stop",
        ),
        risk_constraint=RiskConstraintConfig(
            enabled=risk,
            evaluation="distribution",
            mode=mode,
            lambda_count=2,
            max_candidates=3,
        ),
    )
    _, fixed = runner.build_vehicle_dynamics(base)
    bank = STModelBank.from_model(fixed, (0.4, 0.8))
    coefficient = jnp.asarray([0.9, 0.1], dtype=fixed.params.dtype)
    ensemble = STEnsemble(bank, coefficient, "ensemble_coefficients")
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = jnp.asarray([pose[0], pose[1], 0.05, 10, pose[2], 0, 0], dtype=fixed.params.dtype)
    planner = lmppi.LMPPI(config, ensemble, track, value_input_mode="state_action")
    planner.precompile_hulls()
    baseline_executables = dict(planner.hull_evaluator.compiled)
    monkeypatch.setattr(
        lmppi,
        "predict_value",
        lambda snapshot, states, *args: jnp.zeros(len(states), dtype=states.dtype),
    )

    def select(terminals, speed):
        points = terminals[0, :5] + np.random.default_rng(17).normal(size=(35, 5)) * [
            4,
            1,
            0.2,
            3,
            0.2,
        ]
        return np.column_stack((points, np.zeros((35, 3)))), np.zeros((2, 7))

    risk_config = replace(base.risk, input_mode="state_action", hidden_dims=8, layers=1)
    snapshot = RiskSnapshot(
        initial_state(risk_config, jax.random.key(7)),
        Normalization(
            jnp.zeros(8, dtype=fixed.params.dtype), jnp.ones(8, dtype=fixed.params.dtype)
        ),
        1,
        "state_action",
    )
    cutoff = {"status": "ready", "generation": 1, "threshold": 1.0}
    value = SimpleNamespace(input_mode="state_action", generation=1)
    lmppi.candidate_predictions.clear_cache()
    lmppi.propose_with_constraints.clear_cache()
    chance_probabilities.clear_cache()
    cache_sizes = []
    for generation, probabilities in enumerate(((0.2, 0.8), (0.7, 0.3), (0.5, 0.5)), 1):
        belief = BeliefSnapshot(
            bank,
            coefficient,
            generation,
            "ensemble_fit",
            jnp.asarray(probabilities, dtype=fixed.params.dtype),
        )
        planner.scenario_provider = (
            STBankScenarioProvider.from_snapshot(belief) if chance or risk else None
        )
        frenet = np.asarray([0, 0, 0.05, 10, 0, 0, 0], dtype=np.float32)
        result = planner.update(
            state,
            frenet,
            value,
            SimpleNamespace(select=select),
            capture_debug=True,
            risk_snapshot=snapshot if risk else None,
            risk_cutoff=cutoff if risk else None,
        )
        assert np.isfinite(result.action).all()
        assert len(result.lambdas) == 2 + 2 * chance + risk
        if chance:
            assert result.metrics["probabilistic_safety_status"] == "active"
            assert result.metrics["belief_generation"] == generation
            assert result.metrics["boundary_probability"] <= 0.5
            assert result.metrics["probability_source"] == "categorical_posterior"
        if risk:
            assert result.metrics["risk_constraint_status"] == "active"
            assert result.metrics["risk_evaluation"] == "distribution"
            assert planner.actions.shape == (4, 2)
        else:
            assert planner.actions.shape == (3, 2)  # mean_value needs no u_H.
        np.testing.assert_array_equal(result.model_weights, coefficient)
        assert set(planner.hull_evaluator.batch_sizes) <= {16, 4, 3}
        assert planner.hull_evaluator.sizes == ((10, 20, 30) if hull_query == "facets" else (64,))
        if hull_query == "implicit":
            assert result.metrics["implicit_hull_invalid"] == 0
            assert result.metrics["implicit_hull_queries"] >= 16
            assert result.metrics["hull_points"] == 35
        assert all(
            planner.hull_evaluator.compiled[key] is executable
            for key, executable in baseline_executables.items()
        )
        cache_sizes.append(
            (
                lmppi.candidate_predictions._cache_size(),
                lmppi.propose_with_constraints._cache_size(),
                chance_probabilities._cache_size(),
            )
        )
    assert cache_sizes[0] == cache_sizes[1] == cache_sizes[2]


def test_stop_saves_incomplete_attempt_without_applying_control_or_labeling_failure(
    tmp_path, monkeypatch
):
    config, captured = setup_run(tmp_path, monkeypatch, max_steps=30, max_laps=3)
    config = replace(
        config,
        planner=replace(
            config.planner, probabilistic_safety=ProbabilisticSafetyConfig(enabled=True, fallback="stop")
        ),
    )
    existing = runner.LMPPI

    class Stop(existing):
        def update(self, *args, **kwargs):
            raise SafetyStop("no_feasible_candidate", feasible_candidates=0)

    monkeypatch.setattr(runner, "LMPPI", Stop)
    result = runner.run(config)
    assert result.status == "probabilistic_safety_no_feasible_candidate"
    assert result.trials == result.laps == 2 and result.failures == 0
    assert len(captured["noise"]) == 24  # No 25th action.
    incomplete = load_trajectories(result.directory, "incomplete")
    assert len(incomplete) == 1 and len(incomplete[0].controls) == 0
    assert incomplete[0].termination_reason == result.status
    assert captured["simulator_closed"]
    events = [
        json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()
    ]
    assert any(event["event"] == "probabilistic_safety_stop" for event in events)
    assert not any(event["event"] == "error" for event in events)


def test_sit_fallback_applies_control_and_prints_final_candidate_probabilities(
    tmp_path, monkeypatch, capsys
):
    config, captured = setup_run(tmp_path, monkeypatch, max_steps=30, max_laps=3)
    config = replace(
        config,
        planner=replace(config.planner, probabilistic_safety=ProbabilisticSafetyConfig(
            enabled=True, fallback="baseline",
        )),
    )
    existing = runner.LMPPI

    class Conservative(existing):
        def update(self, *args, **kwargs):
            result = super().update(*args, **kwargs)
            return replace(result, metrics={
                "probabilistic_safety_status": "no_feasible_candidate",
                "chance_fallback": True,
                "chance_fallback_policy": "sit_conservative",
                "boundary_probability": 0.2,
                "boundary_probability_upper": 0.3,
                "boundary_probability_threshold": 0.05,
                "terminal_hull_probability": 0.8,
                "terminal_hull_probability_upper": 0.9,
                "terminal_hull_probability_threshold": 0.05,
                "joint_feasible_candidates": 0,
            })

    monkeypatch.setattr(runner, "LMPPI", Conservative)
    result = runner.run(config)
    assert result.status == "step_limit"
    assert result.learned_steps == 6 and len(captured["noise"]) == 30
    events = [json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()]
    fallbacks = [event for event in events if event["event"] == "probabilistic_safety_fallback"]
    assert len(fallbacks) == 6 and all(event["policy"] == "sit_conservative" for event in fallbacks)
    assert all(event["metrics"]["terminal_hull_probability"] == 0.8 for event in fallbacks)
    assert not any(event["event"] == "probabilistic_safety_stop" for event in events)
    output = capsys.readouterr().out
    assert "P_violate(boundary)=0.2 (upper=0.3, limit=0.05)" in output
    assert "P_violate(hull)=0.8 (upper=0.9, limit=0.05)" in output
    assert captured["simulator_closed"]


def test_independent_checkpoint_allows_changed_thresholds_and_restores_provider(
    tmp_path, monkeypatch
):
    config, captured = setup_run(tmp_path, monkeypatch, max_steps=25, max_laps=4)
    monkeypatch.setattr(runner, "ValueTrainer", ValueTrainer)
    existing = runner.LMPPI
    providers = []

    class Probe(existing):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.config, self.scenario_provider = config, kwargs["scenario_provider"]
            providers.append(self)

    monkeypatch.setattr(runner, "LMPPI", Probe)
    config = replace(
        config,
        planner=replace(
            config.planner,
            belief=BeliefConfig(enabled=False),
            model_bank=ModelBankConfig((0.3, 0.6)),
            probabilistic_safety=ProbabilisticSafetyConfig(enabled=False),
        ),
        value=replace(
            config.value,
            hidden_dims=8,
            layers=1,
            bins=4,
            batch_size=16,
            epochs=1,
            boundary_samples=0,
            samples=2,
        ),
        output=replace(config.output, checkpoint_every_trials=2),
    )
    first = runner.run(config)
    portable = tmp_path / "portable"
    shutil.copytree(first.directory / "checkpoints" / "trial-000002", portable)
    first.directory.rename(tmp_path / "source-unavailable")
    resumed = replace(
        config,
        simulation=replace(config.simulation, max_steps=37, max_laps=3),
        planner=replace(
            config.planner,
            probabilistic_safety=ProbabilisticSafetyConfig(
                enabled=True,
                boundary_threshold=0.2,
                terminal_hull_threshold=0.3,
                fallback="baseline",
            ),
        ),
        output=replace(config.output, resume=str(portable), directory=str(tmp_path / "resumed")),
    )
    continued = runner.run(resumed)
    assert continued.trials == 3 and continued.directory != first.directory
    provider = providers[-1].scenario_provider
    assert provider.metadata["probability_source"] == "declared_bank_prior"
    np.testing.assert_allclose(provider.probabilities, [0.5, 0.5])
    saved = json.loads(
        (continued.directory / "checkpoints" / "trial-000003" / "state.json").read_text()
    )
    assert saved["trials"] == 3
    resolved = yaml.safe_load((continued.directory / "config.yaml").read_text())
    assert resolved["planner"]["probabilistic_safety"]["boundary_threshold"] == 0.2
    assert resolved["planner"]["probabilistic_safety"]["terminal_hull_threshold"] == 0.3
    assert captured["simulator_closed"]


@pytest.mark.parametrize("backend,threshold", [("model_bank", 0.05), ("ensemble_fit", 1.0)])
def test_real_gym_with_active_probability_safety(tmp_path, backend, threshold, float32_runtime):
    from real_gym_fixture import real_gym_config

    config = real_gym_config(tmp_path, backend, "ensemble")
    config = replace(
        config,
        simulation=replace(config.simulation, max_laps=4),
        planner=replace(
            config.planner,
            # One reference lap can leave fewer than six local hull points
            # after a valid policy/order change. Keep enough geometric support
            # while still exercising one complete learned-control attempt.
            safe_set_size=2,
            probabilistic_safety=ProbabilisticSafetyConfig(
                enabled=True,
                boundary_threshold=threshold,
                terminal_hull_threshold=threshold,
                lambda_count=4,
                max_candidates=16,
                fallback="stop",
            ),
        ),
    )
    result = runner.run(config)
    assert result.training_generations >= 1 and result.laps >= config.planner.safe_set_size + 1
    assert result.status in ("lap_limit", "probabilistic_safety_no_feasible_candidate")
    events = [
        json.loads(line) for line in (result.directory / "events.jsonl").read_text().splitlines()
    ]
    if result.status == "lap_limit":
        assert result.trials == config.simulation.max_laps and result.learned_steps > 0
        trace = json.loads((result.directory / "trace.json").read_text())
        metrics = [row["metrics"] for row in trace if row["mode"] == "lmppi"]
        assert metrics and all(row["probabilistic_safety_status"] == "active" for row in metrics)
        assert all(
            row["boundary_probability"] <= threshold
            and row["terminal_hull_probability"] <= threshold
            for row in metrics
        )
    else:
        stopped = [event for event in events if event["event"] == "probabilistic_safety_stop"]
        assert len(stopped) == 1 and stopped[0]["feasible_candidates"] == 0
        assert stopped[0]["probability_source"] == (
            "fitted_predictive_distribution" if backend == "model_bank" else "categorical_posterior"
        )
        assert load_trajectories(result.directory, "incomplete")
    assert not any(event["event"] == "error" for event in events)
