"""Attempt-boundary Bayesian function learning; neural features stay unchanged."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist

from dyna_dist.inference.online import GaussianEvidence, RecalibratableGaussianEvidence
from dyna_dist.inference.posterior import FunctionSamples, components
from dyna_dist.models.features import prior_moments_and_variance
from sit_lmpc.data.dynamics import (
    DynamicsReplay, ReplaySelection, TransitionBatch, initial_transitions, trajectory_transitions,
)
from sit_lmpc.data.trajectories import OUTCOMES, load_trajectories
from sit_lmpc.dynamics.ensemble import weighted_states


@jax.jit
def observation_features(model, states, controls, targets):
    base, basis = components(model, states, controls)
    return basis, model.targets(states, controls, targets) - base


@dataclass(frozen=True)
class GaussianBeliefSnapshot:
    mean: np.ndarray
    factor: np.ndarray
    generation: int = 0


class GaussianBNNLearner:
    """One latent for one unchanged plant, not a new population member per lap.

    Ledger identities make repeated observation of an attempt a no-op. Statistics
    retain every accepted non-held-out transition, not a biased recent/history
    subsample. Posterior inference uses CPU float64 and fixed-shape neural features.
    """

    def __init__(self, config, model, simulation, directory, randomness, emit=lambda *a, **k: None):
        self.config, self.template, self.simulation = config, model, simulation
        self.directory, self.randomness, self.emit = Path(directory), randomness, emit
        prior = jax.device_get(prior_moments_and_variance(model.functions.model.parameters,
                                                         model.functions.model.config))
        self.calibrated = config.posterior_method != "standard"
        evidence_type = RecalibratableGaussianEvidence if self.calibrated else GaussianEvidence
        self.evidence = evidence_type.create(*prior)
        self.snapshot = GaussianBeliefSnapshot(*self.evidence.moments(config.likelihood_power))
        self.active_power = config.likelihood_power
        self.active_method = config.posterior_method
        self.active_variance = self.evidence.variance.copy()
        self.calibration = TransitionBatch.concatenate(())
        self.calibration_status = "warmup" if self.calibrated else "disabled"
        self.processed, self.pending, self.history = {}, {}, []
        self.initial = None
        self.validation = TransitionBatch.concatenate(())

    @property
    def generation(self):
        return self.snapshot.generation

    def predictor(self, fixed_model=None):
        del fixed_model
        dtype = self.template.params.dtype
        model = self.template.replace(
            posterior_mean=jnp.asarray(self.snapshot.mean, dtype=dtype),
            posterior_factor=jnp.asarray(self.snapshot.factor, dtype=dtype),
            belief_generation=jnp.asarray(self.generation, dtype=jnp.int32),
            likelihood_power=jnp.asarray(self.active_power, dtype=dtype),
            posterior_method=self.active_method,
        )
        return model.resample(self.randomness.next_key())

    def _features(self, batch):
        size, dtype = self.config.update_batch_size, self.template.params.dtype
        for start in range(0, len(batch), size):
            count = min(size, len(batch) - start)
            arrays = []
            for values, width in ((batch.states, 7), (batch.controls, 2), (batch.next_states, 7)):
                padded = np.zeros((size, width), dtype=np.dtype(dtype))
                padded[:count] = values[start:start + count]
                arrays.append(jnp.asarray(padded))
            basis, residual = jax.device_get(observation_features(self.template.functions.model, *arrays))
            yield np.asarray(basis[:count], np.float64), np.asarray(residual[:count], np.float64)

    def warmup(self, state):
        batch = TransitionBatch(np.asarray(state)[None], np.zeros((1, 2)),
            np.asarray(state)[None], np.asarray([self.simulation.timestep]),
            np.asarray([self.simulation.integrator_timestep]), np.zeros((1, 2), dtype=int))
        list(self._features(batch))

    def score(self, batch):
        squared = np.zeros(7)
        scale = np.asarray(self.template.functions.model.normalizer.target_scale)
        for basis, residual in self._features(batch):
            error = (residual - basis @ self.snapshot.mean) * scale
            for component in (2, 4, 6):
                error[:, component] = np.arctan2(np.sin(error[:, component]), np.cos(error[:, component]))
            squared += np.sum(error**2, axis=0)
        if not np.isfinite(squared).all():
            return {"status": "nonfinite_prediction", "transitions": len(batch)}
        return {"status": "pre_update" if len(batch) else "no_valid_transitions",
                "transitions": len(batch),
                "one_step_xy_rmse": float(np.sqrt(squared[:2].sum() / len(batch))) if len(batch) else None,
                "component_rmse": np.sqrt(squared / len(batch)).tolist() if len(batch) else None}

    @staticmethod
    def _digest(batch):
        digest = hashlib.sha256()
        for name in TransitionBatch.__dataclass_fields__:
            digest.update(np.ascontiguousarray(getattr(batch, name)).tobytes())
        return digest.hexdigest()

    def _stage(self, identity, batch):
        identity, digest = str(identity), self._digest(batch)
        if identity in self.processed or identity in self.pending:
            old = self.processed.get(identity, self.pending.get(identity, (None, None))[0])
            if digest != old:
                raise ValueError(f"dynamics attempt {identity} changed after observation")
            return False
        if len(batch) and (not np.allclose(batch.intervals, self.template.timestep, rtol=1e-5)
                           or len(np.unique(batch.ids, axis=0)) != len(batch)):
            raise ValueError("BNN evidence needs unique, aligned transitions at the trained interval")
        self.pending[identity] = (digest, batch)
        return True

    def observe(self, trajectory):
        batch = trajectory_transitions(trajectory, self.simulation.integrator_timestep)
        fresh = self._stage(trajectory.attempt_id, batch)
        score = self.score(batch) if fresh else {"status": "already_observed", "transitions": len(batch)}
        self.emit("dynamics_observed", attempt_id=trajectory.attempt_id,
                  outcome=trajectory.outcome, valid_transitions=len(batch),
                  rejected_transitions=len(trajectory.controls) - len(batch),
                  belief_generation=self.generation, predictive=score)
        return score

    def _split(self, batch):
        training, validation = [], []
        for episode in np.unique(batch.ids[:, 0]):
            indices = np.flatnonzero(batch.ids[:, 0] == episode)
            # A held-out suffix is never included in accumulated evidence.
            count = min(len(indices) - 1, int(np.ceil(len(indices) * self.config.validation_fraction)))
            training.extend(indices[:-count] if count else indices)
            if count:
                validation.extend(indices[-count:])
        return batch.take(np.asarray(training, dtype=int)), batch.take(np.asarray(validation, dtype=int))

    def _evaluate(self, snapshot, batch, key, variance=None):
        if not len(batch):
            return {"status": "no_held_out_transitions", "horizons": {}}
        nll = 0.0
        for basis, residual in self._features(batch):
            error = residual - basis @ snapshot.mean
            matrix = basis @ snapshot.factor
            covariance = matrix @ matrix.swapaxes(-1, -2) + np.diag(
                self.active_variance if variance is None else variance)
            chol = np.linalg.cholesky(covariance)
            standardized = np.linalg.solve(chol, error[..., None])[..., 0]
            nll += float(np.sum(standardized**2) + 2 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum()
                         + error.size * np.log(2 * np.pi)) / 2
        if not np.isfinite(nll):
            raise FloatingPointError("nonfinite held-out predictive likelihood")
        replay = DynamicsReplay()
        for episode in np.unique(batch.ids[:, 0]):
            replay.add(int(episode), "held_out", batch.take(np.flatnonzero(batch.ids[:, 0] == episode)))
        selection = ReplaySelection(batch, np.full(len(batch), 1 / len(batch)), len(batch), 0)
        windows = replay.windows(selection, 15)
        indices = np.arange(len(windows))
        if len(indices) > self.config.validation_max_windows:
            indices = indices[np.linspace(0, len(indices) - 1, self.config.validation_max_windows).astype(int)]
        count = len(indices)
        # Keep diagnostic rollouts at a fixed padded shape across every attempt.
        padded = np.pad(indices, (0, self.config.validation_max_windows - count), mode="edge")
        dtype = self.template.params.dtype
        mean, factor = jnp.asarray(snapshot.mean, dtype), jnp.asarray(snapshot.factor, dtype)
        coefficients = mean + jnp.matmul(
            jax.random.normal(key, (self.config.validation_particles, len(mean)), dtype=dtype),
            factor.T, precision=jax.lax.Precision.HIGHEST)
        functions = FunctionSamples(self.template.functions.model, coefficients)
        paths = np.asarray(functions.rollout(jnp.asarray(windows.states[padded], dtype),
                                             jnp.asarray(windows.controls[padded], dtype)))[:count]
        finite = np.where(windows.mask[indices, None, :], np.isfinite(paths).all(axis=-1), True).all(axis=-1)
        result = {"status": "ready" if finite.all() else "nonfinite_rollouts", "windows": count,
                  "transitions": len(batch), "finite_fraction": float(finite.mean()),
                  "nll_normalized_per_component": nll / (7 * len(batch)), "horizons": {},
                  "coverage_definition": "95% coherent whole-XY-path tube; observation noise excluded; not a safety certificate"}
        if not finite.all():
            return result
        point = np.asarray(weighted_states(jnp.asarray(paths).swapaxes(0, 1), functions.weights))
        for horizon in (1, 5, 10, 15):
            valid = windows.mask[indices, :horizon].all(axis=1)
            if not valid.any():
                continue
            truth = windows.targets[indices[valid], :horizon, :2]
            sample, center = paths[valid, :, :horizon, :2], point[valid, :horizon, :2]
            tube = np.quantile(np.linalg.norm(sample - center[:, None], axis=-1).max(axis=-1),
                               0.95, axis=1, method="higher")
            distance = np.linalg.norm(truth - center, axis=-1).max(axis=-1)
            energy = []
            for draws, target in zip(sample, truth, strict=True):
                draws, target = draws.reshape(len(draws), -1) / np.sqrt(horizon), target.reshape(1, -1) / np.sqrt(horizon)
                energy.append(float(cdist(draws, target).mean() - 0.5 * cdist(draws, draws).mean()))
            result["horizons"][str(horizon)] = {
                "windows": int(valid.sum()),
                "endpoint_xy_rmse": float(np.sqrt(np.mean(np.sum((center[:, -1] - truth[:, -1])**2, axis=-1)))),
                "path_coverage_95": float(np.mean(distance <= tube)),
                "path_radius_95": float(tube.mean()), "energy_score": float(np.mean(energy)),
            }
        return result

    def update(self, predictive=None):
        if not self.pending:
            return False
        started = time.perf_counter()
        try:
            return self._update(predictive, started)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            record = {"generation": self.generation, "status": "rejected_nonfinite_update",
                      "attempt_ids": list(self.pending), "reason": str(error),
                      "training_transitions": self.evidence.count,
                      "update_seconds": time.perf_counter() - started}
            if self.calibrated:
                self.calibration_status = "held_nonfinite_update"
                record.update(status=self.calibration_status, published=False,
                              posterior_method=self.config.posterior_method)
            self.processed.update({identity: value[0] for identity, value in self.pending.items()})
            self.pending.clear()
            self.history.append(record)
            self.emit("dynamics_posterior_training", **record)
            self.save()
            return False

    def _update(self, predictive, started):
        if self.calibrated:
            return self._update_calibrated(predictive, started)
        candidate = self.evidence.copy()
        validation, added = [], 0
        for _, batch in self.pending.values():
            training, held_out = self._split(batch)
            for basis, residual in self._features(training):
                candidate.add(basis, residual)
            validation.append(held_out)
            added += len(training)
        validation = TransitionBatch.concatenate(validation)
        mean, factor = candidate.moments(self.config.likelihood_power)
        snapshot = GaussianBeliefSnapshot(mean, factor, self.generation + int(added > 0))
        key = self.randomness.next_key()
        before = self._evaluate(self.snapshot, validation, key)
        after = self._evaluate(snapshot, validation, key)
        accepted = after["status"] != "nonfinite_rollouts"
        if accepted:
            self.evidence, self.snapshot, self.validation = candidate, snapshot, validation
        identities = list(self.pending)
        self.processed.update({identity: record[0] for identity, record in self.pending.items()})
        self.pending.clear()
        record = {"generation": self.generation, "status": "ready" if accepted else "rejected_nonfinite_rollouts",
                  "model_fingerprint": self.template.functions.model.fingerprint,
                  "attempt_ids": identities, "new_training_transitions": added,
                  "training_transitions": self.evidence.count, "held_out_transitions": len(validation),
                  "likelihood_power": self.config.likelihood_power,
                  "inference": "conditional_bayes" if self.config.likelihood_power == 1 else "generalized_bayes",
                  "latent_covariance_trace_ratio": float(np.sum(self.snapshot.factor**2) / np.sum(candidate.factor**2)),
                  "pre_update_prediction": predictive, "validation_before": before, "validation_after": after,
                  "update_seconds": time.perf_counter() - started}
        self.history.append(record)
        self.emit("dynamics_posterior_training", **record,
                  heldout_xy15_rmse=after["horizons"].get("15", {}).get("endpoint_xy_rmse"),
                  heldout_coverage15=after["horizons"].get("15", {}).get("path_coverage_95"))
        if any(metrics["path_coverage_95"] < .9 for metrics in after["horizons"].values()):
            self.emit("dynamics_posterior_warning", generation=self.generation,
                      reason="low_observed_coverage", message="Held-out function coverage below 90%; posterior probabilities are not calibrated safety guarantees")
        self.save()
        return accepted and bool(added)

    def _update_calibrated(self, predictive, started, *, refresh=False):
        from .bnn_calibration import split_calibrated, select_update

        candidate = self.evidence.copy()
        calibration, validation, added, gaps = [self.calibration], [], 0, 0
        for _, batch in self.pending.values():
            training, held_calibration, held_test, discarded = split_calibrated(
                batch, self.config, self.randomness.seed)
            for basis, residual in self._features(training):
                candidate.add(basis, residual)
            calibration.append(held_calibration)
            validation.append(held_test)
            added += len(training)
            gaps += len(discarded)
        calibration = TransitionBatch.concatenate(calibration)
        validation = TransitionBatch.concatenate(validation) if validation else self.validation
        # Keep accepted finite evidence even when its candidate posterior is not
        # published. Later attempts can recalibrate all observations exactly once.
        self.evidence, self.calibration, self.validation = candidate, calibration, validation
        key = self.randomness.next_key()
        decision = select_update(self.template.functions.model, candidate, calibration,
                                 self.snapshot, self.config, key)
        accepted = decision.mean is not None and (added > 0 or refresh)
        before = self._evaluate(self.snapshot, validation, key)
        snapshot = (GaussianBeliefSnapshot(decision.mean, decision.factor, self.generation + 1)
                    if accepted else self.snapshot)
        after = self._evaluate(snapshot, validation, key,
                               decision.variance if accepted else self.active_variance)
        # A nonfinite *prediction* is invalid independently of test target values;
        # finite test error/coverage never participates in choosing a candidate.
        if after["status"] == "nonfinite_rollouts":
            accepted = False
            decision.diagnostics["status"] = "held_nonfinite_prediction"
        if accepted:
            self.snapshot = snapshot
            self.active_variance, self.active_power = decision.variance, decision.power
            self.active_method = self.config.posterior_method
        self.calibration_status = decision.diagnostics["status"]
        identities = list(self.pending)
        self.processed.update({identity: record[0] for identity, record in self.pending.items()})
        self.pending.clear()
        record = {
            "generation": self.generation, "status": self.calibration_status,
            "posterior_method": self.config.posterior_method,
            "active_posterior_method": self.active_method,
            "model_fingerprint": self.template.functions.model.fingerprint,
            "attempt_ids": identities, "new_training_transitions": added,
            "training_transitions": candidate.count, "calibration_transitions": len(calibration),
            "held_out_transitions": len(validation), "new_gap_transitions": gaps,
            "likelihood_power": self.active_power,
            "effective_likelihood_variance": self.active_variance.tolist(),
            "inference": "calibrated_working_likelihood",
            "latent_covariance_trace_ratio": float(np.sum(self.snapshot.factor**2) / np.sum(candidate.factor**2)),
            "calibration": decision.diagnostics, "published": accepted,
            "pre_update_prediction": predictive, "validation_before": before, "validation_after": after,
            "update_seconds": time.perf_counter() - started,
        }
        self.history.append(record)
        self.emit("dynamics_posterior_training", **record,
                  heldout_xy10_rmse=after["horizons"].get("10", {}).get("endpoint_xy_rmse"),
                  heldout_coverage10=after["horizons"].get("10", {}).get("path_coverage_95"))
        self.save()
        return accepted

    def import_initial(self, initial):
        batch = initial_transitions(initial, self.simulation.timestep, self.simulation.integrator_timestep)
        if not len(batch):
            self.emit("dynamics_init_skipped", reason="warm-up has no known aligned physical transitions")
            return
        ids = batch.ids.copy()
        ids[:, 0] = np.where(ids[:, 0] >= 0, -ids[:, 0] - 1, ids[:, 0])
        batch = replace(batch, ids=ids)
        self.initial = batch
        self._stage("initial", batch)
        self.update()

    def save(self):
        if self.calibrated:
            return self._save_calibrated()
        arrays = {"gram": self.evidence.gram, "information": self.evidence.information,
                  "mean": self.snapshot.mean, "factor": self.snapshot.factor}
        for prefix, batch in (("initial", self.initial), ("validation", self.validation)):
            if batch is not None:
                arrays.update({f"{prefix}_{name}": getattr(batch, name) for name in TransitionBatch.__dataclass_fields__})
        np.savez_compressed(self.directory / "dynamics_posterior.npz", **arrays)
        state = {"schema": 1, "model_fingerprint": self.template.functions.model.fingerprint,
                 "feature_dtype": str(self.template.params.dtype),
                 "config": asdict(self.config), "generation": self.generation,
                 "training_transitions": self.evidence.count, "processed": self.processed,
                 "history": self.history, "initial_present": self.initial is not None}
        (self.directory / "dynamics_posterior.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")

    def restore(self, checkpoint, fallback_substep):
        checkpoint = Path(checkpoint)
        state_file = checkpoint / "dynamics_posterior.json"
        if self.calibrated or (state_file.is_file() and
                json.loads(state_file.read_text()).get("schema") == 2):
            return self._restore_extended(checkpoint, fallback_substep)
        rebuild = True
        if state_file.is_file():
            state = json.loads(state_file.read_text())
            if state.get("schema") != 1:
                raise ValueError("unsupported BNN posterior checkpoint")
            rebuild = (state["model_fingerprint"] != self.template.functions.model.fingerprint
                       or state.get("feature_dtype") != str(self.template.params.dtype)
                       or state["config"]["validation_fraction"] != self.config.validation_fraction)
            with np.load(checkpoint / "dynamics_posterior.npz", allow_pickle=False) as arrays:
                if state["initial_present"]:
                    self.initial = TransitionBatch(**{name: arrays[f"initial_{name}"].copy()
                                                      for name in TransitionBatch.__dataclass_fields__})
                if not rebuild:
                    self.evidence.gram = arrays["gram"].copy()
                    self.evidence.information = arrays["information"].copy()
                    self.evidence.count = state["training_transitions"]
                    self.validation = TransitionBatch(**{name: arrays[f"validation_{name}"].copy()
                                                         for name in TransitionBatch.__dataclass_fields__})
            self.history = state["history"]
            generation = state["generation"]
            if not rebuild:
                self.processed = state["processed"]
                generation += int(state["config"]["likelihood_power"] != self.config.likelihood_power)
            self.snapshot = GaussianBeliefSnapshot(*self.evidence.moments(self.config.likelihood_power), generation)
        if rebuild and self.initial is not None:
            self._stage("initial", self.initial)
        # Old frozen/analytical checkpoints can enable posterior learning using
        # their own committed archive. No external experiment is consulted.
        trajectories = sorted((t for outcome in OUTCOMES for t in load_trajectories(checkpoint, outcome)),
                              key=lambda t: t.attempt_id)
        for trajectory in trajectories:
            batch = trajectory_transitions(trajectory, fallback_substep)
            self._stage(trajectory.attempt_id, batch)
        if self.pending:
            self.update()
        self.save()
        self.emit("dynamics_posterior_restored", generation=self.generation,
                  training_transitions=self.evidence.count, rebuilt=rebuild,
                  likelihood_power=self.config.likelihood_power)

    def _save_calibrated(self):
        arrays = {
            "gram_components": self.evidence.gram_components,
            "information_components": self.evidence.information_components,
            "mean": self.snapshot.mean, "factor": self.snapshot.factor,
            "effective_variance": self.active_variance,
        }
        for prefix, batch in (("initial", self.initial), ("validation", self.validation),
                              ("calibration", self.calibration)):
            if batch is not None:
                arrays.update({f"{prefix}_{name}": getattr(batch, name)
                               for name in TransitionBatch.__dataclass_fields__})
        np.savez_compressed(self.directory / "dynamics_posterior.npz", **arrays)
        state = {
            "schema": 2, "model_fingerprint": self.template.functions.model.fingerprint,
            "feature_dtype": str(self.template.params.dtype), "config": asdict(self.config),
            "generation": self.generation, "training_transitions": self.evidence.count,
            "processed": self.processed, "history": self.history,
            "initial_present": self.initial is not None,
            "active_power": self.active_power, "active_method": self.active_method,
            "calibration_status": self.calibration_status,
            "partition_seed": self.randomness.seed,
        }
        (self.directory / "dynamics_posterior.json").write_text(
            json.dumps(state, indent=2, allow_nan=False) + "\n")

    def _restore_extended(self, checkpoint, fallback_substep):
        state_file = checkpoint / "dynamics_posterior.json"
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        if state and state.get("schema") not in (1, 2):
            raise ValueError("unsupported BNN posterior checkpoint")
        saved_config = state.get("config", {})
        partition_fields = ("validation_fraction", "calibration_fraction", "calibration_horizon",
                            "calibration_split")
        # Checkpoints predating random-window sampling used contiguous blocks.
        saved_config = {"calibration_split": "contiguous", **saved_config}
        reuse = (self.calibrated and state.get("schema") == 2
                 and state["model_fingerprint"] == self.template.functions.model.fingerprint
                 and state.get("feature_dtype") == str(self.template.params.dtype)
                 and all(saved_config.get(k) == getattr(self.config, k) for k in partition_fields)
                 and (self.config.calibration_split == "contiguous"
                      or state.get("partition_seed") == self.randomness.seed))
        generation = state.get("generation", 0)
        self.history = state.get("history", [])
        if state:
            with np.load(checkpoint / "dynamics_posterior.npz", allow_pickle=False) as arrays:
                if state["initial_present"]:
                    self.initial = TransitionBatch(**{name: arrays[f"initial_{name}"].copy()
                                                      for name in TransitionBatch.__dataclass_fields__})
                if reuse:
                    for name in ("gram_components", "information_components"):
                        setattr(self.evidence, name, arrays[name].copy())
                    self.evidence.count = state["training_transitions"]
                    self.snapshot = GaussianBeliefSnapshot(arrays["mean"].copy(), arrays["factor"].copy(), generation)
                    self.active_variance = arrays["effective_variance"].copy()
                    self.active_power = state["active_power"]
                    self.active_method = state.get("active_method", saved_config["posterior_method"])
                    self.calibration_status = state["calibration_status"]
                    d, k = len(self.evidence.mean), len(self.evidence.variance)
                    for value, shape in ((self.evidence.gram_components, (k, d, d)),
                                         (self.evidence.information_components, (k, d)),
                                         (self.snapshot.mean, (d,)), (self.snapshot.factor, (d, d)),
                                         (self.active_variance, (k,))):
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError("invalid calibrated BNN checkpoint arrays")
                    if np.any(self.active_variance <= 0) or not 0 < self.active_power <= 1:
                        raise ValueError("invalid calibrated BNN checkpoint likelihood")
                    for prefix in ("calibration", "validation"):
                        setattr(self, prefix, TransitionBatch(**{
                            name: arrays[f"{prefix}_{name}"].copy()
                            for name in TransitionBatch.__dataclass_fields__}))
                    self.processed = state["processed"]
        if not reuse:
            self.snapshot = replace(self.snapshot, generation=generation)
            if self.initial is not None:
                self._stage("initial", self.initial)
        trajectories = sorted((t for outcome in OUTCOMES
                               for t in load_trajectories(checkpoint, outcome)),
                              key=lambda t: t.attempt_id)
        if not reuse and state.get("training_transitions", 0):
            available = {str(t.attempt_id) for t in trajectories} | {"initial"}
            missing = set(state.get("processed", {})) - available
            if missing:
                raise ValueError(f"Rebuilding BNN evidence requires checkpoint trajectories; missing attempts: {sorted(missing)}")
        for trajectory in trajectories:
            self._stage(trajectory.attempt_id, trajectory_transitions(trajectory, fallback_substep))
        inference_fields = ("posterior_method", "likelihood_power", "calibration_powers",
                            "calibration_min_windows", "calibration_target_coverage",
                            "calibration_score_slack", "validation_particles", "validation_max_windows")
        recalibrate = reuse and json.dumps([saved_config.get(k) for k in inference_fields]) != json.dumps(
            [getattr(self.config, k) for k in inference_fields])
        if self.pending:
            self.update()
        elif recalibrate:
            self._update_calibrated(None, time.perf_counter(), refresh=True)
        self.save()
        self.emit("dynamics_posterior_restored", generation=self.generation,
                  training_transitions=self.evidence.count, rebuilt=not reuse,
                  posterior_method=self.config.posterior_method,
                  likelihood_power=self.active_power)
