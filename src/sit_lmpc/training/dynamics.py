"""Synchronous attempt-boundary dynamics learning, independent of risk/value fits."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import jax.numpy as jnp
import numpy as np

from sit_lmpc.data.dynamics import DynamicsReplay, TransitionBatch, trajectory_transitions, initial_transitions
from sit_lmpc.data.trajectories import OUTCOMES, load_trajectories
from sit_lmpc.dynamics.bank import STModelBank
from sit_lmpc.dynamics.belief import BeliefBackend, BeliefSnapshot, LikelihoodBackend
from sit_lmpc.dynamics.ensemble_fit import EnsembleFitBackend
from sit_lmpc.dynamics.model_bank import ModelBankBackend
from sit_lmpc.dynamics.st import PARAMETER_FIELDS
from sit_lmpc.dynamics.ensemble import STEnsemble

BACKENDS = {"model_bank": ModelBankBackend, "ensemble_fit": EnsembleFitBackend,
            "likelihood": LikelihoodBackend}
WINDOW_BACKENDS = ("model_bank", "ensemble_fit")


class DynamicsBeliefLearner:
    def __init__(
        self, config, model, frictions, directory, randomness, emit=lambda *args, **kwargs: None,
        *, prediction_horizon=10,
    ):
        self.config, self.directory, self.randomness, self.emit = (
            config,
            Path(directory),
            randomness,
            emit,
        )
        bank = STModelBank.from_model(model, frictions)
        self.fit_horizon = config.fit_horizon or prediction_horizon
        self.backend: BeliefBackend = BACKENDS[config.backend](bank, config, horizon=self.fit_horizon)
        prior = jnp.asarray(self.backend.prior, dtype=model.params.dtype)
        self.snapshot = BeliefSnapshot(
            bank, prior, 0, config.backend,
            prior if config.backend == "ensemble_fit" else None,
        )
        self.replay = DynamicsReplay()
        self.history = []
        self.last_fit = {"status": "prior_only"}
        self.last_observed_attempt = None

    @property
    def generation(self):
        return self.snapshot.generation

    def observe(self, trajectory):
        batch = trajectory_transitions(trajectory, self.snapshot.bank.integrator_timestep)
        predictive = self.backend.score(batch, self.snapshot.weights)
        self.replay.add(trajectory.attempt_id, trajectory.outcome, batch)
        self.last_observed_attempt = trajectory.attempt_id
        self.emit(
            "dynamics_observed",
            attempt_id=trajectory.attempt_id,
            outcome=trajectory.outcome,
            valid_transitions=len(batch),
            rejected_transitions=len(trajectory.controls) - len(batch),
            belief_generation=self.generation,
            predictive=predictive,
        )
        return predictive

    def update(self, predictive=None):
        selection = self.replay.select(
            self.randomness.host, self.config.history_samples, self.config.recent_fraction
        )
        if self.config.backend in WINDOW_BACKENDS:
            selection = replace(selection, windows=self.replay.windows(selection, self.fit_horizon))
        next_generation = max([self.generation, *(item["generation"] for item in self.history)]) + 1
        snapshot, metrics = self.backend.fit(selection, next_generation)
        if snapshot is not None:
            self.snapshot = snapshot
        weights = np.asarray(self.snapshot.weights)
        record = {
            "generation": self.generation,
            "attempt_id": self.last_observed_attempt,
            "frictions": np.asarray(
                self.snapshot.bank.params[:, PARAMETER_FIELDS.index("mu")]
            ).tolist(),
            "weights": weights.tolist(),
            "map_friction": float(self.snapshot.representative().params[0]),
            "predictor": self.config.predictor,
            "backend": self.config.backend,
            "method": self.backend.method,
            "weight_kind": self.snapshot.weight_kind,
            "posterior_weights": None if self.snapshot.categorical_weights is None
            else np.asarray(self.snapshot.categorical_weights).tolist(),
            "fit_rmse": metrics.get("ensemble_standardized_rmse"),
            "energy_score": metrics.get("energy_score"),
            "recent_transitions": selection.recent_count,
            "history_transitions": selection.history_count,
            "fit": metrics,
            "pre_update_prediction": predictive,
        }
        self.history.append(record)
        self.last_fit = metrics
        self.emit(
            "dynamics_training",
            status=metrics["status"],
            **{key: value for key, value in record.items() if key != "fit"},
        )
        self.save()
        return snapshot is not None

    def predictor(self, fixed_model):
        if self.config.predictor == "ensemble":
            return STEnsemble(self.snapshot.bank, self.snapshot.weights, self.snapshot.weight_kind)
        if self.config.predictor == "map" and self.generation > 0:
            return self.snapshot.representative()
        return fixed_model

    def import_initial(self, initial):
        """Read only known consecutive warm-up transitions; never invent actions."""
        if not initial.controls and getattr(initial, "transitions", None) is None:
            self.emit(
                "dynamics_init_skipped", reason="warm-up snapshot has no aligned physical controls"
            )
            return
        batch = initial_transitions(initial, self.snapshot.bank.timestep,
                                    self.snapshot.bank.integrator_timestep)
        # The bank archive keeps imported warm-up data in one reserved episode.
        # Preserve gaps so horizon fitting never traverses disjoint sections.
        original_ids = batch.ids.copy()
        ids = np.zeros((len(batch), 2), dtype=np.int64)
        ids[:, 0] = -1
        offset = 0
        for index in range(len(batch)):
            if index and (original_ids[index, 0] != original_ids[index - 1, 0]
                          or original_ids[index, 1] != original_ids[index - 1, 1] + 1):
                offset += 1
            ids[index, 1] = index + offset
        batch = replace(batch, ids=ids)
        self.replay.add(-1, "incomplete", batch)
        self.update()

    def save(self):
        bank = self.snapshot.bank
        payload = {
            "params": np.asarray(bank.params),
            "weights": np.asarray(self.snapshot.weights),
            "selection": self.replay.last_selection,
        }
        if self.snapshot.categorical_weights is not None:
            payload["posterior_weights"] = np.asarray(self.snapshot.categorical_weights)
        if -1 in self.replay.episodes:
            batch = self.replay.episodes[-1][1]
            payload.update(
                {f"initial_{name}": getattr(batch, name) for name in batch.__dataclass_fields__}
            )
        np.savez_compressed(self.directory / "dynamics_belief.npz", **payload)
        state = {
            "schema": 1,
            "backend": self.config.backend,
            "weight_kind": self.snapshot.weight_kind,
            "posterior_available": self.snapshot.categorical_weights is not None,
            "method": self.backend.method,
            "fit_horizon": self.fit_horizon if self.config.backend in WINDOW_BACKENDS else None,
            "config": asdict(self.config),
            "generation": self.generation,
            "prior": self.backend.prior.tolist(),
            "timestep": bank.timestep,
            "integrator_timestep": bank.integrator_timestep,
            "attempt_ids": sorted(self.replay.episodes),
            "last_observed_attempt": self.last_observed_attempt,
            "last_fit": self.last_fit,
            "history": self.history,
        }

        # Failure diagnostics may contain nonfinite numerical scores; keep JSON portable.
        def finite(value):
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        (self.directory / "dynamics_belief.json").write_text(
            json.dumps(finite(state), indent=2, allow_nan=False) + "\n"
        )

    def restore(self, checkpoint, fallback_substep):
        checkpoint = Path(checkpoint)
        state_file = checkpoint / "dynamics_belief.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else None
        if state is not None and (
            state.get("schema") != 1 or state.get("backend") not in BACKENDS
        ):
            raise ValueError("unsupported saved dynamics-belief backend")
        allowed = None if state is None else set(state["attempt_ids"])
        for outcome in OUTCOMES:
            for trajectory in load_trajectories(checkpoint, outcome):
                self.replay.add(
                    trajectory.attempt_id,
                    outcome,
                    trajectory_transitions(trajectory, fallback_substep),
                )
        changed = state is None
        if state is not None:
            with np.load(checkpoint / "dynamics_belief.npz", allow_pickle=False) as data:
                if -1 in allowed:
                    self.replay.add(
                        -1,
                        "incomplete",
                        TransitionBatch(
                            **{
                                name: data[f"initial_{name}"].copy()
                                for name in TransitionBatch.__dataclass_fields__
                            }
                        ),
                    )
                if not allowed.issubset(self.replay.episodes):
                    raise ValueError("checkpoint dynamics replay attempts are missing")
                params = data["params"].copy()
                current = np.asarray(self.snapshot.bank.params)
                same_support = params.shape == current.shape and np.allclose(
                    params, current, rtol=1e-6, atol=1e-7
                )
                # Precision, batching, and predictor choice do not invalidate probabilities.
                inference_fields = (
                    "prior",
                    "residual_scales",
                    "recent_fraction",
                    "history_samples",
                    "max_standardized_rmse",
                )
                if self.config.backend != "model_bank":
                    inference_fields += ("evidence_strength",)
                if self.config.backend in WINDOW_BACKENDS:
                    inference_fields += ("fit_horizon", "fit_max_iterations", "fit_tolerance")
                if self.config.backend == "model_bank":
                    inference_fields += ("mean_fit_slack",)
                current_config = asdict(self.config)
                changed = (
                    set(self.replay.episodes) != allowed
                    or not same_support
                    or state["backend"] != self.config.backend
                    # Old model_bank checkpoints contain likelihood weights, not
                    # constrained-energy weights. Refit instead of relabeling them.
                    or (self.config.backend == "model_bank"
                        and state.get("method") != self.backend.method)
                    or (self.config.backend in WINDOW_BACKENDS
                        and state.get("fit_horizon") != self.fit_horizon)
                    or any(
                        json.dumps(state["config"].get(name, current_config[name]))
                        != json.dumps(current_config[name])
                        for name in inference_fields
                    )
                )
                posterior = None
                if self.config.backend == "ensemble_fit":
                    if changed:
                        posterior = jnp.asarray(self.backend.prior, dtype=self.snapshot.bank.params.dtype)
                    elif "posterior_weights" in data:
                        posterior = jnp.asarray(data["posterior_weights"], dtype=self.snapshot.bank.params.dtype)
                self.snapshot = BeliefSnapshot(
                    self.snapshot.bank,
                    jnp.asarray(
                        self.backend.prior if changed else data["weights"],
                        dtype=self.snapshot.bank.params.dtype,
                    ),
                    0 if changed else int(state["generation"]),
                    self.config.backend,
                    posterior,
                )
                if self.config.backend in WINDOW_BACKENDS and not changed:
                    self.backend.weights = np.array(self.snapshot.weights, dtype=float, copy=True)
                    self.backend.weights /= self.backend.weights.sum()
                self.replay.last_selection = data["selection"].copy()
            self.history = state["history"]
            self.last_fit = state["last_fit"]
            self.last_observed_attempt = state["last_observed_attempt"]
        if changed and self.replay.episodes:
            self.emit(
                "dynamics_belief_refit", reason="new learner or changed support/inference settings"
            )
            self.update()
        else:
            self.save()
        self.emit(
            "dynamics_belief_restored",
            generation=self.generation,
            weights=np.asarray(self.snapshot.weights).tolist(),
        )
