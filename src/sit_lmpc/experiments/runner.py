"""ST-only racecar workflow: collect laps, learn values, then use LMPPI."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
from sit_lmpc.config.resume import resume_fixed_fields
from sit_lmpc.data.laps import save_laps
from sit_lmpc.data.trajectories import (
    AttemptCollector,
    TrajectoryDataset,
    load_trajectory,
    load_trajectories,
)
from sit_lmpc.data.experiment_checkpoint import (
    load_checkpoint,
    load_checkpoint_config,
    resume_directory,
    restore_checkpoint,
    save_checkpoint,
)
from sit_lmpc.dynamics import STModel, predictor_step as step
from sit_lmpc.dynamics.ensemble import predictor_context, predictor_friction
from sit_lmpc.dynamics.bank import STModelBank
from sit_lmpc.dynamics.distribution import STBankScenarioProvider
from sit_lmpc.track import Track, map_path
from sit_lmpc.planners.mppi import MPPI
from sit_lmpc.planners.lmppi import (
    LMPPI,
    value_query_inputs,
    endpoint_query_inputs,
    sampled_risk_diagnostic,
)
from sit_lmpc.planners.safe_set import SafeSet
from sit_lmpc.planners.probabilistic import SafetyStop, configured_candidate_counts
from sit_lmpc.training.value import ValueTrainer
from sit_lmpc.training.risk import RiskTrainer
from sit_lmpc.training.risk import Episode
from sit_lmpc.training.dynamics import DynamicsBeliefLearner
from sit_lmpc.training.safe_set_density import SafeSetDensityTrainer
from sit_lmpc.data.checkpoints import load_risk, save_value, cast_snapshot
from sit_lmpc.diagnostics.events import Events
from sit_lmpc.diagnostics.insight import InsightRecorder
from sit_lmpc.diagnostics.insight_context import InsightContexts
from sit_lmpc.diagnostics.native_insight import GymInsight, draw_frame
from sit_lmpc.diagnostics.lap_metrics import LambdaStatistics, summarize_lap
from sit_lmpc.diagnostics.model_comparison import ModelComparisons
from sit_lmpc.diagnostics.keyboard import KeyboardControls, interactive_available
from sit_lmpc.experiments.randomness import ExperimentRandomness, resolve_seed
from sit_lmpc.data.initialization import load_initial, save_initial, restore_safe_set


@dataclass(frozen=True)
class RunResult:
    directory: Path
    status: str
    laps: int
    learned_steps: int
    training_generations: int
    failures: int = 0
    trials: int = 0


def output_directory(config):
    config.validate()
    if config.output.directory:
        directory = Path(config.output.directory).expanduser().resolve()
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty run directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        name = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
        directory = (
            Path(config.output.root).expanduser().resolve()
            / Path(config.simulation.map).name
            / f"{config.output.exp_name}-{name}"
        )
        # No random suffix: refuse a same-name/timestamp collision, even if
        # empty, rather than mixing two automatic experiments' files.
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"Run directory already exists: {directory}. Choose another exp_name or start again after the timestamp advances."
            ) from error
    (directory / "config.yaml").write_text(yaml.safe_dump(config.as_dict()))
    write_metadata(directory)
    return directory


def write_metadata(directory, *, resumed_from=None, checkpoint_trial=None):
    metadata = {
        "schema": 1,
        "package": "sit-lmpc-remastered",
        "version": version("sit-lmpc-remastered"),
        "devices": [str(device) for device in jax.devices()],
        "dependencies": {
            name: version(name)
            for name in ("jax", "jaxlib", "flax", "distrax", "optax", "orbax-checkpoint")
        },
    }
    if resumed_from is not None:
        metadata["resumed_from_checkpoint"] = str(resumed_from)
        metadata["checkpoint_trial"] = checkpoint_trial
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_vehicle_dynamics(config):
    """Separate actual Gym parameters from the assumed prediction model."""
    simulator_params = vehicle_parameters(config.simulation.friction, config.simulation.mass_ratio)
    predictor_params = vehicle_parameters(config.planner.friction, config.planner.mass_ratio)
    return simulator_params, STModel.from_vehicle(predictor_params, config.simulation)


def validate_resume_config(config, saved, checkpoint=None):
    """Only map, failure-label geometry, and saved network formats must match."""
    requested, original = config.as_dict(), saved.as_dict()
    for section, fields in resume_fixed_fields(checkpoint).items():
        for field in fields:
            if requested[section][field] != original[section][field]:
                raise ValueError(
                    f"resume requires the saved experiment configuration for {section}.{field}: "
                    "the checkpoint map, outcome boundary, and network formats must remain compatible"
                )


def restore_actions(saved, template, previous_scale, action_scale):
    """Preserve the overlapping physical controls and zero-pad a changed horizon."""
    actions = jnp.asarray(saved, dtype=template.dtype)
    actions = (actions * jnp.asarray(previous_scale, dtype=template.dtype)
               / jnp.asarray(action_scale, dtype=template.dtype))
    count = min(len(actions), len(template))
    return jnp.zeros_like(template).at[:count].set(jnp.clip(actions[:count], -1, 1))


def validate_resume_checkpoint(directory, checkpoint, state, arrays, tracking_only):
    """Reject incomplete continuation data before archiving any run files."""
    if bool(state["tracking_only"]) != bool(tracking_only):
        raise ValueError("tracking-only mode cannot change during resume")
    counts = state["counts"]
    if state["trials"] != counts["success"] + counts["failure"]:
        raise ValueError("checkpoint trial counts disagree")
    if state["attempts"] != sum(counts.values()):
        raise ValueError("checkpoint attempt counts disagree")
    for name in ("cartesian", "frenet"):
        if arrays[name].shape != (7,) or not np.isfinite(arrays[name]).all():
            raise ValueError(f"checkpoint {name} state is invalid")
    for name in ("value", "risk"):
        if state[f"{name}_generation"] > 0 and not (checkpoint / f"{name}_model").is_dir():
            raise ValueError(f"checkpoint is missing trained {name} model")
    has_constraint_model = (checkpoint / "risk_constraint" / "risk_model").is_dir()
    if state.get("belief_generation") is not None:
        prefix = "dynamics_posterior" if state.get("belief_backend") == "gaussian_bnn" else "dynamics_belief"
        if any(not (checkpoint / f"{prefix}.{extension}").is_file() for extension in ("json", "npz")):
            raise ValueError("checkpoint is missing dynamics-belief state")
        if json.loads((checkpoint / f"{prefix}.json").read_text())["generation"] != state["belief_generation"]:
            raise ValueError("checkpoint dynamics-belief generation differs from trial state")
    if (state.get("risk_constraint_generation") is not None) != has_constraint_model:
        raise ValueError("checkpoint active risk constraint state/model differs")
    if has_constraint_model:
        metadata_path = checkpoint / "risk_constraint" / "risk_metadata.json"
        if not metadata_path.is_file():
            raise ValueError("checkpoint is missing active risk constraint metadata")
        cutoff = (json.loads(metadata_path.read_text()).get("safe_cutoff") or {})
        if (
            cutoff.get("status") != "ready"
            or cutoff.get("generation") != state["risk_constraint_generation"]
            or state["risk_constraint_generation"] > state["risk_generation"]
        ):
            raise ValueError("checkpoint active risk model/cutoff pair differs")
    if len(state["lap_metrics"]) != state["trials"]:
        raise ValueError("checkpoint lap metrics differ from trial count")
    for name, count in (
        ("frenet", state["safe_count"]),
        ("cartesian", state["safe_count"]),
        ("controls", state["safe_control_count"]),
        ("control_masks", state["safe_control_count"]),
    ):
        if any(f"safe_{position}_{name}" not in arrays for position in range(count)):
            raise ValueError(f"checkpoint safe-set {name} is incomplete")
    previous_id = state["safe_previous_attempt_id"]
    if (
        previous_id is not None
        and not (
            directory / "trajectories" / "success" / f"attempt-{previous_id:06d}.npz"
        ).is_file()
    ):
        raise ValueError("checkpoint previous successful attempt is missing")
    for record in state["risk_replay"]:
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory / "trajectories") or not path.is_file():
            raise ValueError(f"checkpoint risk replay file is missing or external: {path}")
        if record["attempt_id"] > state["attempts"]:
            raise ValueError("checkpoint risk replay contains post-checkpoint attempt")


def run(config, tracking_only=False):
    # The listener starts before model initialization. Own it across startup
    # failures as well as the ordinary simulator/training cleanup below.
    with ExitStack() as keyboard_cleanup:
        return _run(config, tracking_only, keyboard_cleanup)


def _run(config, tracking_only, keyboard_cleanup):
    resuming = bool(config.output.resume)
    checkpoint = resume_state = resume_arrays = None
    checkpoint_problems = []
    cutoff_changed = False
    if resuming:
        source_checkpoint = Path(config.output.resume).expanduser().resolve()
        saved_config = load_checkpoint_config(source_checkpoint)
        validate_resume_config(config, saved_config, source_checkpoint)
        config = resolve_seed(config)
        cutoff_changed = (
            config.risk != saved_config.risk
            or config.simulation.seed != saved_config.simulation.seed
            or config.simulation.dtype != saved_config.simulation.dtype
            or config.planner.risk_constraint.enabled != saved_config.planner.risk_constraint.enabled
        )
        checkpoint, resume_state, resume_arrays, checkpoint_problems = load_checkpoint(
            source_checkpoint
        )
        validate_resume_checkpoint(
            checkpoint, checkpoint, resume_state, resume_arrays, tracking_only
        )
        if resume_state.get("sit_terminal_failure"):
            raise ValueError("SIT-LMPC's first physical failure is terminal and cannot be resumed")
        if config.simulation.max_laps <= resume_state["trials"]:
            raise ValueError("increase max_laps beyond the checkpoint's trial count to resume")
        if config.simulation.max_steps <= resume_state["next_step"]:
            raise ValueError("increase max_steps beyond the checkpoint's applied steps to resume")
        requested_directory = (
            config.output.directory
            if config.output.directory != saved_config.output.directory
            else None
        )
        directory = resume_directory(checkpoint, config, requested_directory)
        checkpoint = restore_checkpoint(
            directory, checkpoint, resume_state,
            restore_constraint=not cutoff_changed and config.planner.risk_constraint.enabled,
        )
        config = replace(config, output=replace(config.output, directory=str(directory)))
        persisted = replace(config, output=replace(config.output, resume=None))
        (directory / "config.yaml").write_text(yaml.safe_dump(persisted.as_dict()))
        context_file = directory / "trajectories" / "context.json"
        context = json.loads(context_file.read_text())
        context["config"] = persisted.as_dict()
        context["resumed_from_checkpoint"] = str(source_checkpoint)
        context_file.write_text(json.dumps(context, indent=2) + "\n")
        write_metadata(
            directory, resumed_from=source_checkpoint, checkpoint_trial=resume_state["trials"]
        )
    else:
        config = resolve_seed(config)
    initial_snapshot = None
    if config.output.quick_load and not resuming:
        if tracking_only:
            raise ValueError("tracking-only mode cannot quick-load a learned safe set")
        initial_snapshot = load_initial(
            config.output.init_path, config, config.output.trust_legacy_init
        )
    if not resuming:
        directory = output_directory(config)
    events = Events(directory, config.diagnostics.trace_steps)
    insight = InsightRecorder(directory, config.diagnostics.trace_steps, config, events.emit)
    keyboard = KeyboardControls(config.diagnostics, events.emit)
    keyboard_cleanup.callback(keyboard.close)
    keyboard.start()
    if insight.enabled:
        events.emit("insight_recording", steps=config.diagnostics.trace_steps, file="insight.npz")
    if resuming:
        events.emit(
            "resume",
            checkpoint=str(checkpoint.relative_to(directory)),
            source_checkpoint=str(source_checkpoint),
            skipped_checkpoints=checkpoint_problems,
            next_step=resume_state["next_step"],
            trials=resume_state["trials"],
        )
    simulation, planner = config.simulation, config.planner
    comparison_enabled = (
        insight.enabled or config.diagnostics.keyboard_controls and interactive_available(config.diagnostics)
    )
    track_path = directory / resume_state["map_prefix"] if resuming else map_path(simulation.map)
    track = Track.from_track_path(track_path)
    dataset = TrajectoryDataset(directory, config, float(track.s_frame_max), resume=resuming)
    simulator_params, model = build_vehicle_dynamics(config)
    randomness = ExperimentRandomness(simulation.seed)
    fixed_model = model
    if planner.dynamics.backend == "gaussian_bnn":
        from sit_lmpc.dynamics.gaussian_bnn import load_predictor

        source = planner.dynamics.checkpoint
        if resuming and (
            source is None or source == saved_config.planner.dynamics.checkpoint
        ):
            source = checkpoint / "dynamics_model"
        model = load_predictor(
            planner.dynamics, simulation, randomness.next_key(),
            source=source, destination=directory / "dynamics_model",
        )
    belief_learner = (
        DynamicsBeliefLearner(
            planner.belief, model, planner.model_bank.frictions, directory,
            randomness.fork("dynamics_belief"), events.emit,
            prediction_horizon=planner.horizon,
        ) if planner.belief.enabled and not planner.sit_lmpc_mode else None
    )
    bnn_learning = planner.dynamics.backend == "gaussian_bnn" and planner.dynamics.belief_update == "posterior"
    if bnn_learning:
        from sit_lmpc.training.gaussian_bnn import GaussianBNNLearner

        belief_learner = GaussianBNNLearner(
            planner.dynamics, model, simulation, directory,
            randomness.fork("dynamics_bnn_posterior"), events.emit,
        )
    pending_belief_prediction = None
    if belief_learner is not None:
        model = belief_learner.predictor(fixed_model)
    if planner.dynamics.backend == "gaussian_bnn":
        events.emit("dynamics_model_loaded", **predictor_context(model),
                    training_integrator_timestep=model.integrator_timestep,
                    simulator_integrator_timestep=simulation.integrator_timestep,
                    numerical_substep_mismatch=model.integrator_timestep != simulation.integrator_timestep)

    def current_distribution():
        if not (planner.probabilistic_safety.enabled or (
            planner.risk_constraint.enabled and planner.risk_constraint.evaluation == "distribution"
        )):
            return None
        if planner.dynamics.backend == "gaussian_bnn":
            from sit_lmpc.dynamics.gaussian_bnn import BNNScenarioProvider

            return BNNScenarioProvider(model, planner.dynamics.verification_particles,
                                       planner.dynamics.verification_confidence,
                                       planner.dynamics.chance_estimation)
        if belief_learner is not None:
            return STBankScenarioProvider.from_snapshot(belief_learner.snapshot)
        frictions = (planner.friction,) if planner.sit_lmpc_mode else planner.model_bank.frictions
        bank = STModelBank.from_model(fixed_model, frictions)
        prior = np.ones(len(frictions)) if planner.sit_lmpc_mode or planner.belief.prior is None else np.asarray(planner.belief.prior)
        return STBankScenarioProvider(
            bank, jnp.asarray(prior / prior.sum(), dtype=bank.params.dtype),
            probability_source="fixed_point_mass" if planner.sit_lmpc_mode else "declared_bank_prior",
        )

    nominal = MPPI(planner, model, track, simulation.seed, randomness=randomness)
    learned = LMPPI(
        planner,
        model,
        track,
        simulation.seed,
        config.value.samples,
        randomness=randomness,
        value_input_mode=config.value.input_mode,
        risk_input_mode=config.risk.input_mode if planner.risk_constraint.enabled else None,
        scenario_provider=current_distribution(),
        emit=events.emit,
    )
    learning_context = {
        "master_seed": simulation.seed,
        "map": simulation.map,
        "track_length": float(track.s_frame_max),
        "boundary_half_width": planner.half_width,
        "timestep": simulation.timestep,
        "state_noise": simulation.state_noise,
        "control_noise": list(simulation.control_noise),
        "process_noise": "none; existing noise is controller-input/actuation noise",
        "sit_lmpc_mode": planner.sit_lmpc_mode,
        "dynamics_belief": "gaussian_posterior" if bnn_learning else planner.belief.backend if belief_learner is not None else "disabled",
        "dynamics_predictor": "gaussian_bnn" if planner.dynamics.backend == "gaussian_bnn" else planner.belief.predictor if belief_learner is not None else "fixed",
        "endpoint_action": "sampled u_H; first H controls simulated"
        if (config.value.input_mode == "state_action" and not planner.mean_value)
        or planner.risk_constraint.enabled and config.risk.input_mode == "state_action"
        else "none; state-only estimates",
        "mixed_input_modes": config.risk.enabled
        and config.risk.input_mode != config.value.input_mode,
        "risk_control_use": f"{planner.risk_constraint.evaluation} learned-risk score; independent of geometric chance events",
        "probabilistic_safety": planner.probabilistic_safety.enabled,
        "chance_control_use": "any-step boundary and terminal-hull violation probabilities; LMPPI only, tracking warm-up unconstrained",
        "value_continuation": "logged successful-lap behavior; legacy time target",
        "warmup_source": str(Path(config.output.init_path).resolve())
        if initial_snapshot is not None
        else None,
    }
    safe_set = SafeSet(
        planner,
        simulation.timestep,
        float(track.s_frame_max),
        simulation.seed,
        randomness=randomness,
    )
    density_trainer = (
        SafeSetDensityTrainer(planner.safe_set_density, planner.dynamics, simulation,
            track.s_frame_max, planner.safe_set_size, directory,
            randomness.fork("safe_set_density"), events.emit,
            safe_set_min_states=planner.safe_set_min_states, separate_training=True)
        if planner.safe_set_query == "nf_density" and not tracking_only else None
    )

    def publish_density():
        if density_trainer is None:
            return
        learned.density_snapshot = density_trainer.snapshot
        learned.density_calibration = density_trainer.local_calibration
        counts = [planner.samples, *configured_candidate_counts(planner)]
        provider = getattr(learned, "scenario_provider", None)
        if planner.probabilistic_safety.enabled and provider is not None:
            counts += [count*provider.scenario_count for count in tuple(counts)]
            if getattr(provider, "uses_independent_verification", False):
                counts += [count*provider.verification_particles for count in configured_candidate_counts(planner)]
        density_trainer.warmup(counts)
    trainer = (
        None
        if tracking_only
        else ValueTrainer(
            config.value,
            planner.half_width,
            simulation.seed,
            directory,
            events.emit,
            randomness=randomness,
            prediction_batch_size=tuple(
                dict.fromkeys(
                    (
                        planner.samples,
                        *configured_candidate_counts(planner),
                    )
                )
            )
            if config.value.input_mode == "state_action" and not planner.mean_value
            else planner.samples * planner.horizon,
            context=learning_context,
        )
    )
    def risk_query_counts(cutoff):
        counts = []
        if config.diagnostics.keyboard_controls and interactive_available(config.diagnostics):
            counts.append(planner.samples * planner.horizon)
        if planner.risk_constraint.enabled and cutoff.get("status") == "ready":
            distribution = planner.risk_constraint.evaluation == "distribution"
            provider = learned.scenario_provider if distribution else None
            if not distribution or provider is not None:
                multiplier = provider.scenario_count if distribution else 1
                counts += [count * multiplier * planner.horizon
                           for count in (planner.samples, *configured_candidate_counts(planner))]
        return tuple(dict.fromkeys(counts))

    risk_trainer = (
        RiskTrainer(
            config.risk,
            planner.safe_set_size,
            simulation.seed,
            directory,
            events.emit,
            randomness=randomness.fork("risk"),
            context=learning_context,
            plots=config.diagnostics.summary_plots,
            diagnostic_batch_size=risk_query_counts,
            retain_constraint_pair=planner.risk_constraint.enabled,
        )
        if not tracking_only and (
            config.risk.enabled or resuming and (
                resume_state["risk_generation"] > 0 or resume_state["risk_replay"]
            )
        )
        else None
    )
    simulator, plots, restored_plots = None, None, None
    history, renderer = None, None
    insight_contexts = InsightContexts(config, track)
    comparisons = None
    comparison_before_reset = None
    density_diagnostics = None
    if density_trainer is not None:
        from sit_lmpc.diagnostics.density_calibration import DensityCalibrationDiagnostics

        density_diagnostics = DensityCalibrationDiagnostics()
    previous_risk_samples = None
    shown_validation_generation = None
    shown_constraint_status = None
    shown_probability_status = None

    def keyboard_pause():
        # Called only on the main thread, including while a Qt pause pumps events.
        if keyboard.consume_close_request() and restored_plots is not None:
            restored_plots.close()
        return keyboard.should_pause()

    def risk_pause():
        return keyboard_pause() and keyboard.risk_mode()

    def debug_pause():
        return keyboard_pause() and keyboard.poll()[0]

    def comparison_pause():
        return keyboard_pause() and keyboard.comparison_mode()

    def safe_set_cost_pause():
        return keyboard_pause() and keyboard.safe_set_cost_mode()

    def summary_pause():
        keyboard_pause()
        return config.diagnostics.keyboard_controls and keyboard.poll()[1] and not keyboard.has_navigation()

    def show_validation():
        nonlocal shown_validation_generation
        keyboard_pause()
        validation_selected, selection = keyboard.validation_selection()
        if (
            not validation_selected
            or restored_plots is None
        ):
            return
        snapshot = None if risk_trainer is None else risk_trainer.snapshot
        generation = 0 if snapshot is None else snapshot.generation
        marker = (selection, generation)
        if marker == shown_validation_generation:
            return
        shown_validation_generation = marker
        try:
            data = (
                risk_trainer.validation_samples() if snapshot is not None else {
                    "generation": 0,
                    "predictions": np.asarray([]),
                    "labels": np.asarray([], dtype=int),
                    "cutoff": config.risk.manual_cutoff if config.risk.cutoff_mode == "manual" else None,
                    "cutoff_source": config.risk.cutoff_mode,
                    "cutoff_status": "model_not_trained",
                    "message": "Risk model not trained yet" if config.risk.enabled else "Risk learning is disabled",
                }
            )
            if data is not None:
                restored_plots.risk_validation(
                    data, pause=False, keep_paused=keyboard.validation_mode
                )
        except Exception as error:
            events.emit("risk_validation_plot_error", generation=generation, message=str(error))

    def show_risk(diagnostic, step, status_override=None):
        nonlocal previous_risk_samples
        # A new number key may select risk during a different plot's pause.
        keyboard_pause()
        if diagnostic is not None:
            previous_risk_samples = (diagnostic, step)
        elif previous_risk_samples is not None:
            diagnostic, step = previous_risk_samples
        if diagnostic is not None:
            status = None
        elif not config.risk.enabled:
            status = "Risk learning is disabled in the experiment config."
        elif risk_trainer is None or risk_trainer.snapshot is None:
            status = "Risk model not ready: it needs successful and failed trajectories."
        else:
            status = status_override
        try:
            restored_plots.risk_distribution(
                diagnostic,
                step,
                status=status,
                pause=keyboard.should_pause(),
                keep_paused=risk_pause,
                half_width=planner.half_width,
            )
        except Exception as error:
            events.emit("risk_distribution_plot_error", step=step, message=str(error))
            while risk_pause():
                simulator.render()
                time.sleep(0.05)

    laps, collector = [], None
    success_controller_versions = set()
    lap_metrics, step_violations = [], []
    lambda_statistics = LambdaStatistics()
    init_saved = initial_snapshot is not None
    learned_steps, lap_time = 0, 0.0
    status = "step_limit"
    started = time.perf_counter()
    events.emit(
        "start",
        dynamics="gaussian_bnn" if planner.dynamics.backend == "gaussian_bnn" else "dynamic_ST",
        simulator="st",
        seed=simulation.seed,
        tracking_only=tracking_only,
        simulator_friction=simulation.friction,
        simulator_mass_ratio=simulation.mass_ratio,
        simulator_mass_kg=simulator_params["m"],
        predictor_friction=planner.friction if predictor_friction(model) is not None else None,
        predictor_mass_ratio=planner.mass_ratio if planner.dynamics.backend == "st" else None,
        predictor_kind=predictor_context(model)["kind"],
        predictor=predictor_context(model),
        sit_lmpc_mode=planner.sit_lmpc_mode,
        candidate_frictions=list(planner.model_bank.frictions) if planner.dynamics.backend == "st" and (comparison_enabled or belief_learner is not None) else [],
        dynamics_belief_enabled=belief_learner is not None,
        value_input_mode=config.value.input_mode,
        risk_input_mode=config.risk.input_mode if config.risk.enabled and risk_trainer is not None else None,
    )

    def begin_attempt(cartesian, frenet):
        return AttemptCollector(
            sum(dataset.counts.values()) + 1,
            dataset.trials + 1,
            cartesian,
            frenet,
            track.s_frame_max,
            integrator_timestep=simulation.integrator_timestep,
            action_scale=simulation.action_scale,
        )

    def finish_attempt(outcome, reason):
        nonlocal collector, pending_belief_prediction
        trajectory = collector.finish(outcome, reason)
        saved = dataset.add(trajectory)
        collector = None
        lap = trajectory.as_lap()
        metrics = summarize_lap(
            lap,
            trajectory.lap,
            planner.half_width,
            step_violations,
            completed=outcome == "success",
            termination_reason=reason,
        )
        metrics.update(attempt_id=trajectory.attempt_id, outcome=outcome)
        metrics["lambda_summary"] = lambda_statistics.summary()
        lambda_statistics.clear()
        lap_metrics.append(metrics)
        step_violations.clear()
        (directory / "lap_metrics.json").write_text(json.dumps(lap_metrics, indent=2) + "\n")
        events.emit("attempt", file=str(saved.relative_to(directory)), **metrics)
        if metrics["lambda_summary"]["steps"]:
            events.emit("lambda_summary", attempt_id=trajectory.attempt_id,
                        lap=trajectory.lap, outcome=outcome, **metrics["lambda_summary"])
        if belief_learner is not None:
            pending_belief_prediction = belief_learner.observe(trajectory)
            if outcome == "incomplete":
                belief_learner.save()
        if density_trainer is not None:
            density_trainer.observe(trajectory)
        if risk_trainer is not None and outcome in ("success", "failure"):
            risk_trainer.observe(trajectory, saved)
            risk_trainer.start()
        return trajectory, lap, metrics

    def publish_belief():
        nonlocal model, pending_belief_prediction
        if belief_learner is None:
            return
        previous_predictor = predictor_context(model)
        belief_learner.update(pending_belief_prediction)
        pending_belief_prediction = None
        model = belief_learner.predictor(fixed_model)
        nominal.model = learned.model = model
        learned.scenario_provider = current_distribution()
        if comparisons is not None:
            comparisons.set_belief(model, belief_learner.snapshot)
        current_predictor = predictor_context(model)
        if previous_predictor != current_predictor:
            events.emit("dynamics_predictor_changed", generation=belief_learner.generation,
                        predictor=current_predictor["kind"],
                        previous=previous_predictor, current=current_predictor,
                        risk_context="continuation policy changed; reassess cutoff on fresh attempts")

    def write_checkpoint(next_step, state_cartesian, state_frenet, *, reset_from_failure=False):
        """Publish only a complete trial boundary, after both workers finish."""
        if trainer is not None:
            trainer.wait()
        if risk_trainer is not None:
            risk_trainer.wait()
        if belief_learner is not None:
            belief_learner.save()
        if density_trainer is not None:
            density_trainer.save()
        nominal_horizon = np.asarray(
            getattr(nominal, "actions", np.zeros((planner.horizon, 2), dtype=simulation.dtype))
        )
        learned_horizon = np.asarray(
            getattr(learned, "actions", np.zeros((planner.horizon + 1, 2), dtype=simulation.dtype))
        )
        arrays = {
            "cartesian": np.asarray(state_cartesian),
            "frenet": np.asarray(state_frenet),
            "nominal_actions": np.zeros_like(nominal_horizon)
            if reset_from_failure
            else nominal_horizon,
            "learned_actions": np.zeros_like(learned_horizon)
            if reset_from_failure
            else learned_horizon,
        }
        for label, records in (
            ("frenet", safe_set.frenet),
            ("cartesian", safe_set.cartesian),
            ("controls", safe_set.controls),
            ("control_masks", safe_set.control_masks),
        ):
            for position, value in enumerate(records):
                arrays[f"safe_{position}_{label}"] = np.asarray(value)
        if comparisons is not None:
            arrays["comparison_states"] = np.asarray(list(comparisons.states)).reshape(-1, 7)
            arrays["comparison_actions"] = np.asarray(list(comparisons.actions)).reshape(-1, 2)
        previous_id = (
            None
            if safe_set.previous_trajectory is None
            else safe_set.previous_trajectory.attempt_id
        )
        replay = [
            {
                "path": str(record.path.relative_to(directory)),
                "attempt_id": record.attempt_id,
                "outcome": record.outcome,
                "steps": record.steps,
                "validation": record.validation,
                "policies": [list(policy) for policy in record.policies],
            }
            for record in getattr(getattr(risk_trainer, "replay", None), "episodes", ())
        ]
        with events.lock:
            events_bytes = (directory / "events.jsonl").stat().st_size
        state = {
            "schema": 2,
            "trials": dataset.trials,
            "attempts": sum(dataset.counts.values()),
            "counts": dataset.counts.copy(),
            "applied_steps": dataset.applied_steps,
            "next_step": next_step,
            "learned_steps": learned_steps,
            "elapsed": time.perf_counter() - started,
            "lap_metrics": lap_metrics,
            "success_controller_versions": sorted(
                [list(item) for item in success_controller_versions]
            ),
            "safe_count": len(safe_set.frenet),
            "safe_control_count": len(safe_set.controls),
            "safe_previous_attempt_id": previous_id,
            "init_saved": init_saved,
            "value_generation": 0 if trainer is None else trainer.generation,
            "risk_generation": 0 if risk_trainer is None else risk_trainer.generation,
            "belief_generation": None if belief_learner is None else belief_learner.generation,
            "belief_backend": "gaussian_bnn" if bnn_learning else "model_bank" if belief_learner is not None else None,
            "risk_constraint_generation": None
            if risk_trainer is None or getattr(risk_trainer, "constraint_snapshot", None) is None
            else int(risk_trainer.constraint_snapshot.generation),
            "risk_evaluation": None if risk_trainer is None else risk_trainer.evaluation,
            "risk_replay": replay,
            "trace": list(events.trace),
            "comparison_steps": 0 if comparisons is None else comparisons.steps,
            "simulator": None
            if reset_from_failure or not hasattr(simulator, "snapshot")
            else simulator.snapshot(),
            "reset_from_failure": reset_from_failure,
            "tracking_only": tracking_only,
            "sit_terminal_failure": False,
            "events_bytes": events_bytes,
            "risk_sample_metadata": None
            if previous_risk_samples is None
            else {
                "step": int(previous_risk_samples[1]),
                "generation": int(previous_risk_samples[0]["generation"]),
                "input_mode": previous_risk_samples[0]["input_mode"],
            },
        }
        if previous_risk_samples is not None:
            arrays["risk_sample_probabilities"] = np.asarray(
                previous_risk_samples[0]["probabilities"]
            )
            arrays["risk_sample_frenet"] = np.asarray(previous_risk_samples[0]["frenet"])
        saved = save_checkpoint(
            directory,
            dataset.trials,
            state,
            arrays,
            trainer is not None
            and trainer.snapshot is not None
            and (directory / "value_model").is_dir(),
            risk_trainer is not None
            and risk_trainer.snapshot is not None
            and (directory / "risk_model").is_dir(),
            map_source=track_path,
        )
        events.emit(
            "checkpoint_saved", file=str(saved.relative_to(directory)), trials=dataset.trials
        )
        return saved

    try:
        if not tracking_only:
            learned.precompile_hulls(events.emit)
        simulator = F1TenthSimulator(
            simulation, track_path, config.diagnostics.render, simulator_params
        )
        pose = np.asarray(track.frenet_to_cartesian(0, 0, 0))
        initial = np.asarray([pose[0], pose[1], 0, simulation.initial_speed, pose[2], 0, 0])
        if resuming:
            if (
                dataset.counts != resume_state["counts"]
                or dataset.applied_steps != resume_state["applied_steps"]
            ):
                raise ValueError("committed trajectory counts/steps differ from checkpoint")
            if belief_learner is not None:
                belief_learner.restore(checkpoint, saved_config.simulation.integrator_timestep)
                model = belief_learner.predictor(fixed_model)
                nominal.model = learned.model = model
                learned.scenario_provider = current_distribution()
            initial = resume_arrays["cartesian"].copy()
            for position in range(resume_state["safe_count"]):
                safe_set.frenet.append(resume_arrays[f"safe_{position}_frenet"])
                safe_set.cartesian.append(resume_arrays[f"safe_{position}_cartesian"])
            for position in range(resume_state["safe_control_count"]):
                safe_set.controls.append(resume_arrays[f"safe_{position}_controls"])
                safe_set.control_masks.append(resume_arrays[f"safe_{position}_control_masks"])
            if not planner.safe_set_min_states or planner.safe_set_query != "nf_density":
                safe_set.trim()
            previous_id = resume_state["safe_previous_attempt_id"]
            if previous_id is not None:
                previous_path = (
                    directory / "trajectories" / "success" / f"attempt-{previous_id:06d}.npz"
                )
                safe_set.previous_trajectory = load_trajectory(previous_path)
                safe_set.previous = safe_set.previous_trajectory.as_learning_lap(track.s_frame_max)
            nominal.actions = restore_actions(
                resume_arrays["nominal_actions"], nominal.actions,
                saved_config.simulation.action_scale, simulation.action_scale,
            )
            learned.actions = restore_actions(
                resume_arrays["learned_actions"], learned.actions,
                saved_config.simulation.action_scale, simulation.action_scale,
            )
            if trainer is not None:
                if (checkpoint / "value_model").is_dir():
                    trainer.resume(checkpoint)
                    trainer.snapshot = cast_snapshot(trainer.snapshot, model.params.dtype)
                    trainer.state = trainer.snapshot.state
                    save_value(directory, trainer.snapshot, config.value, learning_context)
                    trainer.warmup(trainer.snapshot)
                if trainer.generation != resume_state["value_generation"]:
                    raise ValueError("value model generation differs from checkpoint")
            if risk_trainer is not None:
                if (checkpoint / "risk_model").is_dir():
                    snapshot = cast_snapshot(
                        load_risk(checkpoint, config.risk, simulation.seed), model.params.dtype
                    )
                    risk_trainer.state = snapshot.state
                    risk_trainer.snapshot = snapshot
                    risk_trainer.generation = int(snapshot.generation)
                if risk_trainer.generation != resume_state["risk_generation"]:
                    raise ValueError("risk model generation differs from checkpoint")
                risk_trainer.evaluation = resume_state["risk_evaluation"]
                cutoff = (risk_trainer.evaluation or {}).get("safe_cutoff")
                if cutoff is not None and cutoff.get("generation") != risk_trainer.generation:
                    raise ValueError("risk cutoff generation differs from checkpoint")
                constraint_generation = resume_state.get("risk_constraint_generation")
                if constraint_generation is not None:
                    active_path = checkpoint / "risk_constraint"
                    active_snapshot = cast_snapshot(
                        load_risk(active_path, config.risk, simulation.seed), model.params.dtype
                    )
                    active_metadata = json.loads((active_path / "risk_metadata.json").read_text())
                    active_cutoff = active_metadata["safe_cutoff"]
                    if (
                        active_snapshot.generation != constraint_generation
                        or active_cutoff.get("generation") != constraint_generation
                        or active_cutoff.get("status") != "ready"
                        or constraint_generation > risk_trainer.generation
                    ):
                        raise ValueError("checkpoint active risk model/cutoff pair differs")
                    risk_trainer.constraint_snapshot = active_snapshot
                    risk_trainer.constraint_cutoff = active_cutoff
                replay = []
                for record in resume_state["risk_replay"]:
                    trajectory_path = (directory / record["path"]).resolve()
                    if (
                        not trajectory_path.is_relative_to(directory / "trajectories")
                        or not trajectory_path.is_file()
                    ):
                        raise ValueError(
                            f"risk replay path is outside this experiment: {trajectory_path}"
                        )
                    trajectory = load_trajectory(trajectory_path)
                    if (
                        trajectory.attempt_id != record["attempt_id"]
                        or len(trajectory.controls) != record["steps"]
                    ):
                        raise ValueError(f"risk replay trajectory differs: {trajectory_path}")
                    replay.append(
                        Episode(
                            trajectory_path,
                            record["attempt_id"],
                            record["outcome"],
                            record["steps"],
                            record["validation"],
                            tuple(tuple(policy) for policy in record["policies"]),
                        )
                    )
                risk_trainer.replay.episodes = replay
                if config.risk.enabled and not saved_config.risk.enabled:
                    known = {record.attempt_id for record in replay}
                    historical = sorted(
                        [trajectory for outcome in ("success", "failure")
                         for trajectory in load_trajectories(directory, outcome)],
                        key=lambda trajectory: trajectory.attempt_id,
                    )
                    for trajectory in historical:
                        if trajectory.attempt_id not in known:
                            path = directory / "trajectories" / trajectory.outcome / f"attempt-{trajectory.attempt_id:06d}.npz"
                            risk_trainer.observe(trajectory, path)
                if cutoff_changed:
                    risk_trainer.refresh_cutoff()
            if trainer is not None and safe_set.ready and trainer.snapshot is None:
                trainer.start(safe_set.training_data(config.value.input_mode))
                trainer.wait()
            laps = [trajectory.as_lap() for trajectory in load_trajectories(directory, "success")]
            if len(laps) != resume_state["counts"]["success"]:
                raise ValueError("saved successful lap history differs from checkpoint")
            lap_metrics = resume_state["lap_metrics"]
            success_controller_versions = set(
                map(tuple, resume_state["success_controller_versions"])
            )
            init_saved = bool(resume_state["init_saved"])
            learned_steps = int(resume_state["learned_steps"])
            started = time.perf_counter() - float(resume_state["elapsed"])
            events.trace.extend(resume_state["trace"])
            risk_samples = resume_state.get("risk_sample_metadata")
            if risk_samples is not None:
                previous_risk_samples = (
                    {
                        "probabilities": resume_arrays["risk_sample_probabilities"],
                        "frenet": resume_arrays["risk_sample_frenet"],
                        "generation": risk_samples["generation"],
                        "input_mode": risk_samples["input_mode"],
                    },
                    risk_samples["step"],
                )
        elif initial_snapshot is not None:
            restore_safe_set(safe_set, initial_snapshot)
            if belief_learner is not None:
                belief_learner.import_initial(initial_snapshot)
                model = belief_learner.predictor(fixed_model)
                learned.scenario_provider = current_distribution()
                nominal.model = learned.model = model
            if config.output.quick_load_init:
                initial = initial_snapshot.state.copy()
                # Warm-up snapshots store H executed-horizon controls. Initialize
                # the extra proposal mean to zero; it is sampled during planning.
                learned.actions = learned.actions.at[: planner.horizon].set(
                    jnp.asarray(initial_snapshot.actions, dtype=model.params.dtype)
                )
        if density_trainer is not None:
            density_trajectories = load_trajectories(directory, "success")
            for trajectory in density_trajectories:
                density_trainer.observe(trajectory)
            safe_set.restore_state_ids(density_trajectories)
            if initial_snapshot is not None:
                density_trainer.import_initial(initial_snapshot, safe_set)
            elif resuming:
                density_trainer.restore_initial(checkpoint, safe_set)
            if resuming:
                density_trainer.restore_training_history(checkpoint)
            safe_set.trim()
            restored_density = resuming and density_trainer.restore(checkpoint, safe_set)
            if not restored_density and density_trainer.episodes:
                density_trainer.fit(safe_set)
            if not density_trainer.episodes:
                events.emit("safe_set_density_status", status="warmup",
                    reason="needs_original_success_trajectories; imported extended init arrays are not calibration data")
            publish_density()
        if resuming and resume_state["simulator"] is not None and hasattr(simulator, "restore"):
            (cartesian, frenet), _ = simulator.restore(initial, resume_state["simulator"])
        else:
            (cartesian, frenet), _ = simulator.reset(initial)
        if resuming and not np.allclose(frenet, resume_arrays["frenet"], atol=1e-5):
            raise ValueError("simulator restored a different Frenet state from the checkpoint")
        collector = begin_attempt(cartesian, frenet)
        if abs(frenet[1]) > planner.half_width:
            finish_attempt("failure", "invalid_lap_start")
            raise ValueError("The lap-start state is out of bounds; cannot retry from it")
        if belief_learner is not None:
            if bnn_learning:
                belief_learner.warmup(cartesian)
            else:
                belief_learner.backend.warmup(cartesian)
            belief_learner.save()
        if comparison_enabled:
            # Ground truth belongs only to this diagnostics object, never to
            # MPPI/LMPPI or their current-belief predictor.
            comparison_frictions = (
                (planner.friction,) if planner.sit_lmpc_mode else planner.model_bank.frictions
            )
            comparisons = ModelComparisons(
                model,
                STModel.from_vehicle(simulator_params, simulation),
                comparison_frictions,
                planner.horizon,
                planner.scan_rollout,
            )
            comparisons.warmup(cartesian)
            if belief_learner is not None:
                comparisons.set_belief(model, belief_learner.snapshot)
            if resuming:
                comparisons.states = deque(
                    resume_arrays.get("comparison_states", np.empty((0, 7))),
                    maxlen=planner.horizon + 1,
                )
                comparisons.actions = deque(
                    resume_arrays.get("comparison_actions", np.empty((0, 2))),
                    maxlen=planner.horizon,
                )
                comparisons.steps = int(resume_state["comparison_steps"])
            events.emit(
                "model_bank_warmup",
                seconds=comparisons.warmup_seconds,
                frictions=list(comparison_frictions),
            )
        if initial_snapshot is not None:
            trainer.start(
                safe_set.training_data(config.value.input_mode), epochs=2 * config.value.epochs
            )
            trainer.wait()
            events.emit(
                "quick_load",
                from_file=str(Path(config.output.init_path).resolve()),
                restore_state=config.output.quick_load_init,
                safe_set_laps=len(safe_set.frenet),
                epochs=2 * config.value.epochs,
                compatibility=initial_snapshot.metadata.get("compatibility", "validated"),
            )
        if config.value.input_mode == "state_action" and not tracking_only:
            warm_started = time.perf_counter()
            queries = value_query_inputs(
                jnp.zeros((planner.samples, planner.horizon, 7), dtype=model.params.dtype),
                jnp.asarray(frenet, dtype=model.params.dtype),
                jnp.zeros((planner.samples, planner.horizon + 1, 2), dtype=model.params.dtype),
                model.action_scale,
                planner.mean_value,
            )
            jax.block_until_ready(queries)
            if not planner.mean_value:
                for count in dict.fromkeys(
                    (
                        *configured_candidate_counts(planner),
                    )
                ):
                    jax.block_until_ready(
                        endpoint_query_inputs(
                            jnp.zeros((count, planner.horizon, 7), dtype=model.params.dtype),
                            jnp.zeros((count, planner.horizon + 1, 2), dtype=model.params.dtype),
                            model.action_scale,
                        )
                    )
            events.emit("value_action_queries_warmed", seconds=time.perf_counter() - warm_started)
        if hasattr(nominal, "warmup"):
            nominal.warmup(cartesian, simulation.initial_speed)
        renderer = None
        if config.diagnostics.render:
            from sit_lmpc.diagnostics.rendering import TrajectoryRenderer

            renderer = TrajectoryRenderer(simulator, track, planner.half_width)
        if config.diagnostics.plots:
            from sit_lmpc.diagnostics.plots import LivePlots

            plots = LivePlots()
        if (
            config.diagnostics.summary_plots
            or interactive_available(config.diagnostics)
        ):
            try:
                from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots

                restored_plots = MatplotlibPlots(
                    directory, interactive_available(config.diagnostics)
                )
                restored_plots.keyboard_command = keyboard.focused_command
            except ImportError as error:
                events.emit("matplotlib_unavailable", message=f"Install the plots extra: {error}")
        if config.diagnostics.render and interactive_available(config.diagnostics):
            if hasattr(simulator, "bind_keyboard"):
                simulator.bind_keyboard(keyboard)
            history = GymInsight(insight, simulator, keyboard,
                lambda frame, number: draw_frame(insight, frame, number, restored_plots, track, config),
                renderer=renderer, plots=restored_plots, emit=events.emit)
        if not resuming:
            write_checkpoint(0, cartesian, frenet)
        for index in range(resume_state["next_step"] if resuming else 0, simulation.max_steps):
            if history is not None and history.poll():
                status = "interrupted"
                break
            if not np.isfinite(cartesian).all() or not np.isfinite(frenet).all():
                status = "nonfinite_state"
                break
            if cartesian[3] < 0.1:
                status = "speed_too_low"
                break
            prediction_state = cartesian.copy()
            keyboard_pause()
            show_validation()
            debug_mode, _ = keyboard.poll()
            risk_mode = keyboard.risk_mode()
            comparison_mode = keyboard.comparison_mode()
            safe_set_cost_mode = keyboard.safe_set_cost_mode()
            active_risk_snapshot = None
            active_risk_cutoff = None
            if planner.risk_constraint.enabled and not tracking_only:
                active_risk_snapshot = risk_trainer.constraint_snapshot
                active_risk_cutoff = risk_trainer.constraint_cutoff
                if active_risk_snapshot is None:
                    reason = (
                        "model_not_trained"
                        if risk_trainer.snapshot is None
                        else ((risk_trainer.evaluation or {}).get("safe_cutoff") or {}).get(
                            "status", "cutoff_not_ready"
                        )
                    )
                    constraint_status = ("unavailable", reason, None)
                else:
                    if planner.risk_constraint.evaluation == "distribution" and learned.scenario_provider is None:
                        constraint_status = ("unavailable", "dynamics_distribution_unavailable",
                                             int(active_risk_snapshot.generation))
                        active_risk_snapshot = active_risk_cutoff = None
                    else:
                        latest = risk_trainer.snapshot
                        ready_status = (
                            "active" if latest is None or latest.generation == active_risk_snapshot.generation
                            else "using_previous"
                        )
                        constraint_status = (
                            ready_status if safe_set.ready else "ready_waiting_for_lmppi",
                            None,
                            int(active_risk_snapshot.generation),
                        )
                if constraint_status != shown_constraint_status:
                    events.emit(
                        "risk_constraint_status",
                        status=constraint_status[0],
                        reason=constraint_status[1],
                        generation=constraint_status[2],
                        step=index,
                    )
                    shown_constraint_status = constraint_status
            if safe_set.ready and not tracking_only:
                # Capture one coherent immutable snapshot for the entire update.
                if trainer.snapshot is None:
                    trainer.wait()
                snapshot = trainer.snapshot
                generation = snapshot.generation
                noise = randomness.state_noise(simulation.state_noise)
                risk_snapshot = active_risk_snapshot or (
                    risk_trainer.snapshot
                    if (insight.enabled or (risk_mode or debug_mode) and restored_plots is not None)
                    and risk_trainer is not None
                    else None
                )
                result = learned.update(
                    prediction_state + noise,
                    frenet,
                    snapshot,
                    safe_set,
                    capture_debug=insight.enabled or ((debug_mode or safe_set_cost_mode) and restored_plots is not None),
                    **({"risk_snapshot": risk_snapshot} if risk_snapshot is not None else {}),
                    **({"risk_cutoff": active_risk_cutoff} if active_risk_cutoff is not None else {}),
                    **(
                        {"capture_risk": True}
                        if active_risk_snapshot is not None
                        and risk_mode
                        and restored_plots is not None
                        else {}
                    ),
                )
                mode = "lmppi"
                learned_steps += 1
                if config.diagnostics.print_lambdas:
                    events.emit(
                        "selected_lambdas",
                        step=index,
                        lambdas=dict(zip(
                            result.metrics.get("lambda_terms", ("hull", "boundary")),
                            result.lambdas,
                            strict=True,
                        )),
                    )
                if planner.probabilistic_safety.enabled:
                    probability_status = (result.metrics.get("probabilistic_safety_status", "unavailable"),
                                          result.metrics.get("belief_generation"))
                    if probability_status != shown_probability_status:
                        events.emit("probabilistic_safety_status", status=probability_status[0],
                                    generation=probability_status[1], step=index,
                                    estimator=result.metrics.get("probability_estimator"),
                                    samples=result.metrics.get("sample_count"),
                                    metrics=result.metrics)
                        shown_probability_status = probability_status
                if result.metrics.get("risk_fallback"):
                    events.emit(
                        "risk_no_feasible_candidate",
                        step=index,
                        generation=result.metrics["risk_generation"],
                        cutoff=result.metrics["risk_cutoff"],
                        chosen_risk=result.metrics["risk_score"],
                    )
                if result.metrics.get("chance_fallback"):
                    events.emit("probabilistic_safety_fallback", step=index,
                                status=result.metrics["probabilistic_safety_status"],
                                policy="sit_conservative", metrics=result.metrics)
            else:
                if planner.probabilistic_safety.enabled and shown_probability_status != ("tracking_warmup_unconstrained", None):
                    events.emit("probabilistic_safety_status", status="tracking_warmup_unconstrained",
                                step=index, reason="terminal safe-set/value learner not ready or tracking-only run")
                    shown_probability_status = ("tracking_warmup_unconstrained", None)
                result = nominal.update(prediction_state, simulation.initial_speed)
                mode = "mppi"
                generation = 0
            if planner.dynamics.backend == "gaussian_bnn":
                model = (learned if mode == "lmppi" else nominal).model
                if comparisons is not None:
                    comparisons.belief = model
            if insight.enabled:
                from types import SimpleNamespace

                context = insight_contexts.capture(
                    controller=SimpleNamespace(model=getattr(learned if mode == "lmppi" else nominal, "model", model)),
                    safe_set=safe_set, comparisons=comparisons, density=density_trainer,
                    risk=risk_trainer, risk_snapshot=risk_snapshot if mode == "lmppi" else None,
                    laps=laps, metrics=lap_metrics)
                insight.record(
                    result, step=index, attempt=collector.attempt_id, lap=dataset.trials + 1,
                    time=lap_time, mode=mode, cartesian=cartesian, frenet=frenet,
                    actions=(learned if mode == "lmppi" else nominal).actions,
                    context=context,
                )
            comparison = None
            density_view = None
            if comparison_mode and comparisons is not None and restored_plots is not None:
                try:
                    controller = learned if mode == "lmppi" else nominal
                    comparison = comparisons.planned(
                        cartesian, controller.actions[: planner.horizon], index
                    )
                except Exception as error:
                    events.emit("model_comparison_error", step=index, message=str(error))
            if safe_set_cost_mode and density_diagnostics is not None and restored_plots is not None:
                try:
                    density_view = density_diagnostics.evaluate(
                        result, safe_set, density_trainer.snapshot, density_trainer.validation,
                        planner, density_trainer.local_calibration,
                        training=density_trainer.diagnostics,
                    )
                except Exception as error:
                    events.emit("density_calibration_plot_error", step=index, message=str(error))
                    density_view = dict(status="unavailable", message=f"NF cutoff inspection unavailable: {error}")
            if safe_set_cost_mode and restored_plots is not None:
                diagnostic = None
                if result.debug is not None:
                    try:
                        from sit_lmpc.diagnostics.safe_set_costs import compare_safe_set_costs

                        diagnostic = compare_safe_set_costs(result, safe_set, planner, controller=learned)
                    except Exception as error:
                        events.emit("safe_set_cost_plot_error", step=index, message=str(error))
                try:
                    view = density_view or dict(status="unavailable", message="NF calibration is not ready yet.")
                    view.setdefault("penalty_transform", planner.safe_set_density.penalty_transform)
                    figure = restored_plots.safe_set_costs(diagnostic, index, density=view)
                except Exception as error:
                    events.emit("safe_set_cost_plot_error", step=index, message=str(error))
                    figure = None
                if figure is not None and keyboard.should_pause():
                    restored_plots.wait_for_interaction(figure, safe_set_cost_pause)
                elif keyboard.should_pause():
                    while safe_set_cost_pause():
                        simulator.render()
                        time.sleep(0.05)
            elif risk_mode and restored_plots is not None:
                show_risk(result.risk, index)
            elif comparison is not None:
                try:
                    figure = restored_plots.model_comparison(
                        comparison, track, planner.half_width, show=True,
                        safe_set_states=(np.concatenate(safe_set.cartesian)
                                         if safe_set.cartesian else None),
                    )
                    if keyboard.should_pause():
                        restored_plots.wait_for_interaction(figure, comparison_pause)
                except Exception as error:
                    events.emit("model_comparison_plot_error", step=index, message=str(error))
                    while comparison_pause():
                        simulator.render()
                        time.sleep(0.05)
            elif debug_mode and result.debug is not None and restored_plots is not None:
                # Pause before applying this update's action, not in a worker or mid-step.
                restored_plots.debug(
                    result, index, pause=keyboard.should_pause(), keep_paused=debug_pause
                )
            elif (debug_mode or risk_mode or comparison_mode or safe_set_cost_mode) and keyboard.should_pause():
                # Pausing remains independent of plotting; 0/right Alt releases it.
                events.emit(
                    "keyboard_paused",
                    step=index,
                    message="Press 0 or right Alt to continue; plot unavailable",
                )
                while debug_pause() or risk_pause() or comparison_pause() or safe_set_cost_pause():
                    simulator.render()
                    time.sleep(0.05)
            if not risk_mode and keyboard.risk_mode() and restored_plots is not None:
                # A 3 pressed in an open debug plot inspects this same update.
                risk_diagnostic = result.risk
                status_override = None
                diagnostic_snapshot = active_risk_snapshot or (
                    None if risk_trainer is None else risk_trainer.snapshot
                )
                if (
                    risk_diagnostic is None
                    and result.debug is not None
                    and diagnostic_snapshot is not None
                ):
                    sample_actions = result.debug["sample_actions"]
                    if diagnostic_snapshot.input_mode == "state_action":
                        sample_actions = np.concatenate(
                            (
                                sample_actions,
                                result.debug["sample_endpoint_actions"][:, None],
                            ),
                            axis=1,
                        )
                    risk_diagnostic = sampled_risk_diagnostic(
                        diagnostic_snapshot,
                        result.debug["states"],
                        sample_actions,
                        model.action_scale,
                    )
                elif risk_diagnostic is None:
                    status_override = (
                        "Samples unavailable for this step; advance once to inspect risk."
                    )
                show_risk(risk_diagnostic, index, status_override)
            keyboard_pause()
            show_validation()
            if history is not None and history.poll():
                raise KeyboardInterrupt
            physical_noise = randomness.host.normal(0, simulation.control_noise, 2)
            (next_cartesian, next_frenet), control, terminated, truncated, info = simulator.step(
                result.action, physical_noise
            )
            insight.applied(next_cartesian, next_frenet, control)
            if insight.enabled:
                insight.display(simulator.capture_render() if hasattr(simulator, "capture_render") else None)
            lap_time += simulation.timestep
            collector.append(
                next_cartesian, next_frenet, control, lap_time, mode, generation, index,
                belief_generation=0 if belief_learner is None else belief_learner.generation,
                contact=info.get("collision", False),
            )
            if mode == "lmppi":
                lambda_statistics.observe(result.lambdas,
                    result.metrics.get("lambda_terms", ("hull", "boundary")))
            step_violations.append((result.safe_set_violation, result.boundary_violation))
            if not np.isfinite(next_cartesian).all() or not np.isfinite(next_frenet).all():
                status = "nonfinite_state"
                break
            failure = (
                "out_of_bounds"
                if abs(next_frenet[1]) > planner.half_width
                else "collision"
                if info.get("collision", False)
                else None
            )
            outcome = "failure" if failure else "success" if collector.completed_lap else None
            finished = finish_attempt(outcome, failure or "lap_completed") if outcome else None
            if outcome == "success":
                laps.append(finished[1])
                success_controller_versions.update(
                    zip(
                        finished[0].controllers.tolist(),
                        finished[0].value_generations.tolist(),
                        strict=True,
                    )
                )
            # Enabled trace/live consumers retain the exact applied-control prediction.
            # trace_steps=0 and a non-due/off live plot are the only skip case.
            live_prediction_due = plots is not None and index % config.diagnostics.plot_interval == 0
            predicted = None
            if insight.enabled or live_prediction_due:
                actual_action = jnp.asarray(
                    control / np.asarray(simulation.action_scale), dtype=model.params.dtype
                )
                predicted = np.asarray(
                    jax.device_get(step(model, jnp.asarray(cartesian), actual_action))
                )
                insight.prediction(predicted)
            if outcome == "failure" and insight.enabled:
                try:
                    insight.save(failure, crash=True, materialize=False)
                except Exception as error:
                    events.emit("insight_error", message=str(error))
            if comparisons is not None:
                comparisons.observe(
                    cartesian, control / np.asarray(simulation.action_scale), next_cartesian
                )
            if config.diagnostics.trace_steps > 0:
                events.record(
                    next_cartesian,
                    predicted,
                    result.action,
                    dataset.trials - int(outcome is not None),
                    mode,
                    result.metrics,
                )
            if renderer:
                renderer.update(result, safe_set=safe_set, frenet_state=next_frenet)
            if insight.enabled:
                insight.display(simulator.capture_render() if hasattr(simulator, "capture_render") else None,
                                renderer.snapshot() if renderer is not None else None)
                context.summary = insight_contexts.summary(laps, lap_metrics)
            simulator.render()
            if live_prediction_due:
                plots.update(next_cartesian, predicted, result.action)
            if restored_plots is not None and index % config.diagnostics.plot_interval == 0:
                restored_plots.pump()
            keyboard_pause()
            if history is not None:
                hold = bool(outcome == "failure" and config.diagnostics.keep_open_on_failure)
                if history.poll(hold=hold, reason=failure if outcome == "failure" else None):
                    status = failure if outcome == "failure" else "interrupted"
                    break
            if mode == "lmppi" and learned_steps == 1:
                events.emit("learned_controller_started", step=index, **result.metrics)
            # Key 2 refreshes after every finished trial, including failures.
            if (
                outcome
                and keyboard.poll()[1]
                and restored_plots is not None
            ):
                try:
                    restored_plots.summaries(
                        laps,
                        lap_metrics,
                        track,
                        planner.half_width,
                        show=True,
                        pause=True,
                        keep_paused=summary_pause,
                    )
                except Exception as error:
                    events.emit("summary_plot_error", message=str(error))
            keyboard_pause()
            if outcome is not None:
                publish_belief()
            if outcome == "failure":
                failed, _, _ = finished
                if insight.enabled:
                    try:
                        insight.save(failure, crash=True, pump=simulator.render if history is not None else lambda: None)
                    except Exception as error:
                        events.emit("insight_error", message=str(error))
                if planner.sit_lmpc_mode:
                    status = failure
                    break
                if dataset.trials >= simulation.max_laps:
                    write_checkpoint(
                        index + 1, failed.cartesian[0], failed.frenet[0], reset_from_failure=True
                    )
                    status = "lap_limit"
                    break
                if comparisons is not None:
                    try:
                        replay = comparisons.recorded(index)
                        if replay is not None:
                            comparison_before_reset = replay
                    except Exception as error:
                        events.emit("model_comparison_error", step=index, message=str(error))
                    comparisons.reset_history()
                if risk_trainer is not None:
                    risk_trainer.wait()
                    show_validation()
                (cartesian, frenet), _ = simulator.reset(failed.cartesian[0].copy())
                nominal.actions = jnp.zeros((planner.horizon, 2), dtype=model.params.dtype)
                learned.actions = jnp.zeros_like(learned.actions)
                collector = begin_attempt(cartesian, frenet)
                lap_time = 0.0
                events.emit(
                    "lap_reset",
                    failed_attempt=failed.attempt_id,
                    next_attempt=collector.attempt_id,
                    failures=dataset.counts["failure"],
                )
                if dataset.trials % config.output.checkpoint_every_trials == 0:
                    write_checkpoint(index + 1, cartesian, frenet)
                continue
            if outcome == "success":
                _, lap, metrics = finished
                events.emit(
                    "lap",
                    lap=dataset.trials,
                    successful_laps=len(laps),
                    laptime=lap.laptime,
                    mean_speed=float(lap.cartesian[:, 3].mean()),
                    mode=mode,
                    safe_set_violation=result.safe_set_violation,
                    boundary_violation=result.boundary_violation,
                    lambdas=result.lambdas,
                    lambda_summary=metrics["lambda_summary"],
                    elapsed=time.perf_counter() - started,
                )
                previous_safe_laps = len(safe_set.frenet)
                safe_set_added = safe_set.add_lap(lap, finished[0])
                if planner.safe_set_min_states and len(safe_set.frenet) != previous_safe_laps:
                    events.emit("safe_set_retention", laps=len(safe_set.frenet),
                                states=safe_set.state_count, min_states=planner.safe_set_min_states)
                if safe_set_added and trainer is not None:
                    trainer.context = {
                        **learning_context,
                        "collected_success_controller_versions": sorted(
                            success_controller_versions
                        ),
                        "successful_attempt_count": len(laps),
                    }
                    trainer.start(safe_set.training_data(config.value.input_mode))
                if density_trainer is not None:
                    density_trainer.fit(safe_set)
                    publish_density()
                if safe_set.ready and trainer is not None and not init_saved:
                    save_initial(
                        directory / "init.npz", safe_set, next_cartesian, nominal.actions, config,
                        trajectories=load_trajectories(directory, "success"),
                        density_training=None if density_trainer is None else density_trainer.training_history(),
                    )
                    init_saved = True
                    events.emit("initial_snapshot_saved", file=str(directory / "init.npz"))
                save_laps(directory / "laps.npz", laps)
                (directory / "lap_metrics.json").write_text(
                    json.dumps(lap_metrics, indent=2) + "\n"
                )
                lap_time = 0.0
                if dataset.trials >= simulation.max_laps:
                    write_checkpoint(index + 1, next_cartesian, next_frenet)
                    status = "lap_limit"
                    break
                if dataset.trials % config.output.checkpoint_every_trials == 0:
                    write_checkpoint(index + 1, next_cartesian, next_frenet)
            if terminated or truncated:
                status = "simulator_terminated" if terminated else "simulator_truncated"
                break
            if outcome == "success":
                if risk_trainer is not None:
                    risk_trainer.wait()
                    show_validation()
                collector = begin_attempt(next_cartesian, next_frenet)
            cartesian, frenet = next_cartesian, next_frenet
    except SafetyStop as error:
        status = "probabilistic_safety_" + error.reason
        events.emit("probabilistic_safety_stop", reason=error.reason,
                    policy="stop_before_action", **error.metrics)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "error"
        events.emit("error", type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            stop_reason = status
            if (history is not None and config.diagnostics.keep_open_on_failure
                    and not history.finished
                    and stop_reason not in ("step_limit", "lap_limit", "interrupted")
                    and insight.history
                    and history.last_hold != (insight.history[-1]["step"], stop_reason)):
                try:
                    insight.save(stop_reason, crash=True, materialize=False)
                except Exception as error:
                    events.emit("insight_error", message=str(error))
                history.poll(hold=True, reason=stop_reason)
            if insight.enabled:
                try:
                    insight.save(stop_reason, crash=stop_reason not in ("step_limit", "lap_limit", "interrupted"),
                                 pump=simulator.render if history is not None and simulator.window_open()
                                 and stop_reason != "nonfinite_state" else lambda: None)
                except Exception as error:
                    events.emit("insight_error", message=str(error))
            partial_metrics = None
            training_failure = None
            if trainer is not None:
                try:
                    trainer.wait()
                except Exception as error:
                    status = "training_error"
                    training_failure = error
                    events.emit("training_error", message=str(error))
            if risk_trainer is not None:
                try:
                    risk_trainer.wait()
                    risk_trainer.save_report()
                    show_validation()
                except Exception as error:
                    status = "risk_training_error"
                    training_failure = training_failure or error
                    events.emit("risk_training_error", message=str(error))
            save_laps(directory / "laps.npz", laps)
            if collector is not None and (
                collector.controls or status not in ("lap_limit", "step_limit")
            ):
                _, partial_lap, partial_metrics = finish_attempt("incomplete", stop_reason)
                if len(partial_lap.times):
                    save_laps(directory / "partial_lap.npz", [partial_lap])
                    events.emit("partial_lap", file="partial_lap.npz", **partial_metrics)
            events.save_trace()
            (directory / "lap_metrics.json").write_text(json.dumps(lap_metrics, indent=2) + "\n")
            if comparisons is not None:
                try:
                    # Replay the latest bounded real window, including a failed
                    # terminal step, without advancing or resetting Gym.
                    comparison = (
                        comparisons.recorded(comparisons.steps - 1) or comparison_before_reset
                    )
                    if comparison is not None:
                        if restored_plots is not None:
                            restored_plots.model_comparison(
                                comparison, track, planner.half_width,
                                safe_set_states=(
                                    np.concatenate(safe_set.cartesian) if safe_set.cartesian else None
                                ),
                            )
                except Exception as error:
                    events.emit("model_comparison_error", message=str(error))
            if restored_plots is not None and config.diagnostics.summary_plots:
                try:
                    restored_plots.summaries(laps, lap_metrics, track, planner.half_width)
                except Exception as error:
                    # A graphics failure must not hide the controller's stop reason
                    # or prevent the summary/checkpoint/cleanup from being finalized.
                    events.emit("summary_plot_error", message=str(error))
            summary = {
                "status": status,
                "laps": len(laps),
                "successful_laps": len(laps),
                "trials": dataset.trials,
                "history_laps": 0,
                "attempts": sum(dataset.counts.values()),
                "failed_attempts": dataset.counts["failure"],
                "incomplete_attempts": dataset.counts["incomplete"],
                "applied_steps": dataset.applied_steps,
                "learned_steps": learned_steps,
                "training_generations": 0 if trainer is None else trainer.generation,
                "value_input_mode": config.value.input_mode,
                "safe_set_query": planner.safe_set_query,
                "safe_set_density": None if density_trainer is None else density_trainer.diagnostics,
                "risk_training_generations": 0 if risk_trainer is None else risk_trainer.generation,
                "risk_input_mode": config.risk.input_mode if risk_trainer is not None else None,
                "risk_evaluation": None if risk_trainer is None else risk_trainer.evaluation,
                "dynamics_belief_generation": 0 if belief_learner is None else belief_learner.generation,
                "dynamics_belief_weights": None if belief_learner is None or bnn_learning else np.asarray(belief_learner.snapshot.weights).tolist(),
                "dynamics_posterior": None if not bnn_learning else {
                    "training_transitions": belief_learner.evidence.count,
                    "likelihood_power": planner.dynamics.likelihood_power,
                    "last_update": belief_learner.history[-1] if belief_learner.history else None,
                },
                "predictor_friction": predictor_friction(model),
                "dynamics_predictor": predictor_context(model),
                "probabilistic_safety": config.as_dict()["planner"]["probabilistic_safety"],
                "scenario_provider": None if getattr(learned, "scenario_provider", None) is None
                else learned.scenario_provider.metadata,
                "elapsed": time.perf_counter() - started,
                "partial_lap": (
                    None
                    if partial_metrics is None or not partial_metrics["steps"]
                    else {"file": "partial_lap.npz", **partial_metrics}
                ),
            }
            memory = jax.devices()[0].memory_stats()
            if memory is not None:
                summary["device_memory"] = {
                    key: int(memory[key])
                    for key in (
                        "bytes_in_use",
                        "peak_bytes_in_use",
                        "bytes_reserved",
                        "peak_bytes_reserved",
                        "bytes_limit",
                    )
                    if memory.get(key) is not None
                }
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            events.emit("finish", **summary, directory=str(directory))
            if training_failure is not None:
                raise training_failure
        finally:
            keyboard_cleanup.close()
            if restored_plots is not None:
                restored_plots.close()
            if plots is not None:
                plots.close()
            if simulator is not None:
                simulator.close()
    return RunResult(
        directory,
        status,
        len(laps),
        learned_steps,
        0 if trainer is None else trainer.generation,
        dataset.counts["failure"],
        dataset.trials,
    )
