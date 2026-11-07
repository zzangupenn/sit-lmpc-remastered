"""Experiment-local, balanced outcome learning with sampling-corrected probabilities."""

from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time

import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sit_lmpc.data.normalization import Normalization
from sit_lmpc.data.trajectories import load_trajectory
from sit_lmpc.training.features import features, feature_names, input_metadata, inference_snapshot


class RiskNetwork(nn.Module):
    width: int
    layers: int

    @nn.compact
    def __call__(self, inputs):
        for _ in range(self.layers):
            inputs = nn.silu(nn.Dense(self.width)(inputs))
        return nn.Dense(1)(inputs)[:, 0]


@struct.dataclass
class RiskSnapshot:
    state: TrainState
    normalization: Normalization
    generation: int
    input_mode: str = struct.field(pytree_node=False, default="state")


def initial_state(config, key):
    network = RiskNetwork(config.hidden_dims, config.layers)
    params = network.init(key, jnp.zeros((1, len(feature_names(config.input_mode)))))
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(8), optax.adam(config.learning_rate)),
    )


@jax.jit
def train_step(state, inputs, targets, weights):
    def loss(params):
        logits = state.apply_fn(params, inputs)
        # Divide by batch size, NOT random weight sum: the Horvitz--Thompson
        # weights below target the original uniformly sampled state population.
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets) * weights)

    value, gradients = jax.value_and_grad(loss)(state.params)
    return state.apply_gradients(grads=gradients), value


@jax.jit
def train_epoch(state, inputs, targets, weights):
    def update(state, batch):
        return train_step(state, *batch)

    state, losses = jax.lax.scan(update, state, (inputs, targets, weights))
    return state, losses.mean()


@jax.jit
def predict_risk(snapshot, states, actions=None, physical_failure=None, goal=None):
    inputs = snapshot.normalization.encode_inputs(features(states, snapshot.input_mode, actions))
    risk = jax.nn.sigmoid(snapshot.state.apply_fn(snapshot.state.params, inputs))
    if goal is not None:
        risk = jnp.where(goal, 0.0, risk)
    if physical_failure is not None:
        risk = jnp.where(physical_failure, 1.0, risk)
    return risk


@jax.jit
def terminal_risk(physical_failure, goal):
    """Terminal outcomes need no model or fictional next control."""
    return jnp.where(physical_failure, 1.0, jnp.where(goal, 0.0, jnp.nan))


def trajectory_inputs(trajectory, mode):
    states = trajectory.frenet[:-1].copy()
    # Keep the same five input features while making lap phase explicit in s.
    states[:, 0] = trajectory.frenet[0, 0] + trajectory.progress[:-1]
    return np.asarray(features(states, mode, trajectory.controls))


@dataclass(frozen=True)
class Episode:
    path: Path
    attempt_id: int
    outcome: str
    steps: int
    validation: bool
    policies: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RiskBatchPlan:
    inputs: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    normalization: Normalization
    metadata: dict
    validation: tuple[Episode, ...]


class RiskReplay:
    """Only descriptors are archived in memory; active arrays are bounded by laps."""

    def __init__(self, directory, config, capacity):
        self.directory = Path(directory).resolve()
        self.config, self.capacity = config, capacity
        self.episodes = []

    def add(self, trajectory, path, rng):
        path = Path(path).resolve()
        if not path.is_relative_to(self.directory / "trajectories"):
            raise ValueError("risk replay cannot import another experiment's trajectories")
        if trajectory.outcome == "incomplete" or not len(trajectory.controls):
            return
        if any(record.attempt_id == trajectory.attempt_id for record in self.episodes):
            raise ValueError("risk replay attempt IDs must be unique")
        same = [record for record in self.episodes if record.outcome == trajectory.outcome]
        # Reserve the first example of each outcome for training. Assignment is
        # made once, before any training; held-out episodes can never leak later.
        validation = bool(same) and rng.random() < self.config.validation_fraction
        policies = tuple(
            sorted(
                set(
                    zip(
                        trajectory.controllers.tolist(),
                        trajectory.value_generations.tolist(),
                        strict=True,
                    )
                )
            )
        )
        self.episodes.append(
            Episode(
                path,
                trajectory.attempt_id,
                trajectory.outcome,
                len(trajectory.controls),
                validation,
                policies,
            )
        )

    def counts(self):
        return {
            split: {
                outcome: sum(
                    record.steps
                    for record in self.episodes
                    if record.validation == (split == "validation") and record.outcome == outcome
                )
                for outcome in ("success", "failure")
            }
            for split in ("training", "validation")
        }

    def prepare(self, rng, normalization=None):
        pools = {
            outcome: [e for e in self.episodes if not e.validation and e.outcome == outcome]
            for outcome in ("success", "failure")
        }
        if not all(pools.values()):
            return None
        size = self.config.batch_size
        fail_count = max(1, min(size - 1, round(size * self.config.failure_fraction)))
        batch_counts = {"failure": fail_count, "success": size - fail_count}
        source_count = sum(e.steps for pool in pools.values() for e in pool)
        selected, arrays, class_weights, inclusions = {}, {}, {}, {}
        for outcome, pool in pools.items():
            count = min(self.capacity, len(pool))
            records = [pool[i] for i in rng.choice(len(pool), count, replace=False)]
            selected[outcome] = [e.attempt_id for e in records]
            arrays[outcome] = np.concatenate(
                [
                    trajectory_inputs(load_trajectory(e.path), self.config.input_mode)
                    for e in records
                ]
            )
            inclusion = count / len(pool)
            fraction = batch_counts[outcome] / size
            # Uniform episode subset + uniform states within its concatenation.
            # Inverse inclusion corrects different lap lengths and class caps,
            # in addition to the deliberately balanced batch class fraction.
            class_weights[outcome] = len(arrays[outcome]) / (source_count * fraction * inclusion)
            inclusions[outcome] = inclusion
        if normalization is None:
            inputs = np.concatenate(list(arrays.values()))
            low, high = inputs.min(axis=0), inputs.max(axis=0)
            normalization = Normalization(
                jnp.asarray(np.r_[low, 0.0]), jnp.asarray(np.r_[high, 1.0])
            )
        epochs = self.config.epochs
        batches = self.config.batches_per_epoch
        x = np.empty((epochs, batches, size, len(feature_names(self.config.input_mode))))
        y = np.empty((epochs, batches, size))
        w = np.empty_like(y)
        for epoch in range(epochs):
            for batch in range(batches):
                offset = 0
                for outcome in ("success", "failure"):
                    count = batch_counts[outcome]
                    indices = rng.integers(len(arrays[outcome]), size=count)
                    stop = offset + count
                    x[epoch, batch, offset:stop] = arrays[outcome][indices]
                    y[epoch, batch, offset:stop] = float(outcome == "failure")
                    w[epoch, batch, offset:stop] = class_weights[outcome]
                    offset = stop
        metadata = {
            "state_counts_before_balancing": self.counts(),
            "active_attempts": selected,
            "episode_inclusion_probabilities": inclusions,
            "batch_outcome_counts": batch_counts,
            "example_loss_weights": class_weights,
            "loss_population": "uniform nonterminal states from eligible training episodes",
            "correction": "inverse episode-inclusion and batch state-sampling weights",
            "continuation_scope": "logged behavior-policy mixture; not a frozen-controller guarantee",
            "controller_versions": sorted({policy for e in self.episodes for policy in e.policies}),
            "training_attempts": [e.attempt_id for e in self.episodes if not e.validation],
            "validation_attempts": [e.attempt_id for e in self.episodes if e.validation],
            "s_convention": "lap-start Frenet s plus unwrapped within-attempt progress",
        }
        return RiskBatchPlan(
            x, y, w, normalization, metadata, tuple(e for e in self.episodes if e.validation)
        )


def fit_risk(state, plan, config, generation):
    dtype = plan.normalization.low.dtype
    normalization = jax.tree.map(lambda x: jnp.asarray(x, dtype=dtype), plan.normalization)
    inputs = normalization.encode_inputs(jnp.asarray(plan.inputs, dtype=dtype))
    losses = []
    for epoch in range(config.epochs):
        state, loss = train_epoch(
            state,
            inputs[epoch],
            jnp.asarray(plan.targets[epoch], dtype=dtype),
            jnp.asarray(plan.weights[epoch], dtype=dtype),
        )
        losses.append(float(loss))
    if not np.isfinite(losses).all() or not all(
        np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state.params)
    ):
        raise FloatingPointError("nonfinite risk training loss or parameters")
    return RiskSnapshot(state, normalization, generation, config.input_mode), losses


def calibration_metrics(predictions, targets, bins=10):
    p, y = np.asarray(predictions), np.asarray(targets)
    if not len(p):
        return None
    indices = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        mask = indices == index
        if mask.any():
            rows.append(
                {
                    "low": index / bins,
                    "high": (index + 1) / bins,
                    "count": int(mask.sum()),
                    "predicted": float(p[mask].mean()),
                    "observed": float(y[mask].mean()),
                }
            )
    return {
        "examples": len(p),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "failure_fraction": float(y.mean()),
        "mean_probability": float(p.mean()),
        "reliability": rows,
    }


def safe_risk_cutoff(predictions, targets, episode_ids, target_fraction, min_states, min_episodes):
    """Largest inclusive risk cutoff meeting an empirical held-out success fraction."""
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=int)
    episodes = np.asarray(episode_ids)
    if p.ndim != 1 or p.shape != y.shape or p.shape != episodes.shape:
        raise ValueError("risk cutoff inputs must be aligned one-dimensional arrays")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("risk cutoff needs finite probabilities and binary outcomes")
    success_episodes = len(np.unique(episodes[y == 0]))
    failure_episodes = len(np.unique(episodes[y == 1]))
    result = {
        "status": "insufficient_held_out_outcomes",
        "threshold": None,
        "target_safe_fraction": target_fraction,
        "min_states": min_states,
        "min_success_episodes": min_episodes,
        "held_out_states": len(p),
        "held_out_success_episodes": success_episodes,
        "held_out_failure_episodes": failure_episodes,
        "selected_states": 0,
        "selected_success_states": 0,
        "selected_failure_states": 0,
        "selected_success_episodes": 0,
        "observed_safe_fraction": None,
    }
    if not success_episodes or not failure_episodes:
        return result
    order = np.argsort(p, kind="stable")
    ordered_p, ordered_y, ordered_episodes = p[order], y[order], episodes[order]
    # Endpoints include all equal scores, matching the eventual q <= threshold gate.
    ends = np.r_[np.flatnonzero(np.diff(ordered_p)), len(p) - 1] + 1
    selected_successes = np.cumsum(ordered_y == 0)[ends - 1]
    safe_fractions = selected_successes / ends
    _, first_success_positions = np.unique(ordered_episodes[ordered_y == 0], return_index=True)
    success_positions = np.flatnonzero(ordered_y == 0)[first_success_positions]
    success_episode_counts = np.searchsorted(np.sort(success_positions), ends, side="left")
    supported = (ends >= min_states) & (success_episode_counts >= min_episodes)
    eligible = supported & (safe_fractions >= target_fraction)
    if not eligible.any():
        result["status"] = "target_not_met" if supported.any() else "insufficient_support"
        return result
    selected = np.flatnonzero(eligible)[-1]
    count = int(ends[selected])
    successes = int(selected_successes[selected])
    result.update(
        status="ready",
        threshold=float(ordered_p[count - 1]),
        selected_states=count,
        selected_success_states=successes,
        selected_failure_states=count - successes,
        selected_success_episodes=int(success_episode_counts[selected]),
        observed_safe_fraction=float(safe_fractions[selected]),
    )
    return result


def balanced_cutoff_samples(predictions, targets, episode_ids, seed, generation):
    """Draw equal held-out state counts per outcome without changing Brier inputs."""
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=int)
    episodes = np.asarray(episode_ids)
    success = np.flatnonzero(y == 0)
    failure = np.flatnonzero(y == 1)
    count = min(len(success), len(failure))
    sampling = {
        "method": "equal_outcome_states_without_replacement",
        "available": {"success": len(success), "failure": len(failure)},
        "sampled": {"success": count, "failure": count},
        "seed_source": "experiment master seed and risk generation",
    }
    if count == 0:
        return p[:0], y[:0], episodes[:0], sampling
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(generation), 0x5AFE]))
    selected = np.r_[
        rng.choice(success, count, replace=False),
        rng.choice(failure, count, replace=False),
    ]
    return p[selected], y[selected], episodes[selected], sampling


def evaluate(snapshot, records, batch_size, config, cutoff_seed=0, *, return_samples=False):
    predictions, targets, episode_ids, first_p, first_y = [], [], [], [], []
    for record in records:
        trajectory = load_trajectory(record.path)
        x = trajectory_inputs(trajectory, snapshot.input_mode)
        episode_predictions = []
        for offset in range(0, len(x), batch_size):
            block = x[offset : offset + batch_size]
            padded = np.pad(block, ((0, batch_size - len(block)), (0, 0)), mode="edge")
            padded = jnp.asarray(padded, dtype=snapshot.normalization.low.dtype)
            actions = padded[:, 5:] if snapshot.input_mode == "state_action" else None
            p = np.asarray(predict_risk(snapshot, padded[:, :5], actions))[: len(block)]
            episode_predictions.extend(p.tolist())
        predictions.extend(episode_predictions)
        targets.extend([int(record.outcome == "failure")] * len(x))
        episode_ids.extend([record.attempt_id] * len(x))
        first_p.append(episode_predictions[0])
        first_y.append(int(record.outcome == "failure"))
    both = len(set(first_y)) == 2
    cutoff_p, cutoff_y, cutoff_episodes, cutoff_sampling = balanced_cutoff_samples(
        predictions, targets, episode_ids, cutoff_seed, snapshot.generation
    )
    if config.cutoff_mode == "manual" and not len(cutoff_p):
        # A fixed threshold needs no balancing support; still inspect any held-out data.
        cutoff_p = np.asarray(predictions, dtype=float)
        cutoff_y = np.asarray(targets, dtype=int)
        cutoff_episodes = np.asarray(episode_ids)
        cutoff_sampling = {
            **cutoff_sampling,
            "method": "available_held_out_states_without_both_outcomes",
            "sampled": dict(cutoff_sampling["available"]),
        }
    if config.cutoff_mode == "manual":
        selected = cutoff_p <= config.manual_cutoff
        selected_labels = cutoff_y[selected]
        cutoff = {
            "status": "ready",
            "source": "manual",
            "threshold": float(config.manual_cutoff),
            "held_out_states": len(cutoff_p),
            "selected_states": int(selected.sum()),
            "selected_success_states": int((selected_labels == 0).sum()),
            "selected_failure_states": int((selected_labels == 1).sum()),
            "selected_success_episodes": len(np.unique(cutoff_episodes[selected & (cutoff_y == 0)])),
            "observed_safe_fraction": (
                float((selected_labels == 0).mean()) if selected.any() else None
            ),
        }
    else:
        cutoff = safe_risk_cutoff(
            cutoff_p,
            cutoff_y,
            cutoff_episodes,
            config.target_safe_fraction,
            config.cutoff_min_states,
            config.cutoff_min_episodes,
        )
        cutoff["source"] = "learned"
    cutoff["generation"] = int(snapshot.generation)
    cutoff["sampling"] = cutoff_sampling
    report = {
        "status": "held_out_both_outcomes" if both else "insufficient_held_out_outcomes",
        "states": calibration_metrics(predictions, targets),
        "episode_starts": calibration_metrics(first_p, first_y),
        "safe_cutoff": cutoff,
        "note": (
            "The manual cutoff is fixed by configuration and does not require held-out support. "
            "Brier/reliability and held-out validation samples remain diagnostic."
            if config.cutoff_mode == "manual" else
            "Brier/reliability use all held-out states; the cutoff uses an equally "
            "sampled success/failure subset. Its selected success fraction is an "
            "artificially balanced statistic, not a natural failure probability, "
            "probability recalibration, or safety guarantee."
        ),
    }
    if return_samples:
        return report, {
            "generation": int(snapshot.generation),
            "predictions": cutoff_p,
            "labels": cutoff_y,
            "episode_ids": cutoff_episodes,
            "cutoff": cutoff["threshold"],
            "cutoff_status": cutoff["status"],
            "cutoff_source": cutoff["source"],
        }
    return report


class RiskTrainer:
    def __init__(
        self,
        config,
        capacity,
        seed,
        directory,
        emit=lambda *a, **k: None,
        randomness=None,
        context=None,
        plots=False,
        diagnostic_batch_size=None,
        retain_constraint_pair=False,
    ):
        from sit_lmpc.experiments.randomness import ExperimentRandomness

        self.config, self.seed, self.directory = config, seed, Path(directory)
        self.emit, self.context, self.plots = emit, context, plots
        self.diagnostic_batch_size = diagnostic_batch_size
        self.retain_constraint_pair = retain_constraint_pair
        self.randomness = (
            randomness if randomness is not None else ExperimentRandomness(seed).fork("risk")
        )
        self.replay = RiskReplay(directory, config, config.buffer_size or capacity)
        self.state = initial_state(config, self.randomness.next_key())
        self.snapshot, self.thread, self.error = None, None, None
        self.generation = 0
        self.evaluation = None
        self.validation_plot_data = None
        self.constraint_snapshot = None
        self.constraint_cutoff = None

    def validation_samples(self):
        """Return held-out plot samples for either manual or learned cutoff mode."""
        if self.snapshot is None:
            return None
        if (
            self.validation_plot_data is None
            or self.validation_plot_data["generation"] != self.snapshot.generation
        ):
            records = [record for record in self.replay.episodes if record.validation]
            _, self.validation_plot_data = evaluate(
                self.snapshot,
                records,
                self.config.batch_size,
                self.config,
                self.seed,
                return_samples=True,
            )
        return self.validation_plot_data

    def refresh_cutoff(self):
        """Apply changed resume cutoff settings to the restored model without training."""
        from sit_lmpc.data.checkpoints import save_risk

        self.wait()
        self.constraint_snapshot = self.constraint_cutoff = None
        self.validation_plot_data = None
        if self.snapshot is None:
            return
        records = [record for record in self.replay.episodes if record.validation]
        evaluation, samples = evaluate(
            self.snapshot, records, self.config.batch_size, self.config, self.seed,
            return_samples=True,
        )
        stored = json.loads((self.directory / "risk_metadata.json").read_text())
        metadata = {
            key: value for key, value in stored.items()
            if key not in ("schema", "model", "risk_config", "generation", "safe_cutoff")
        }
        metadata["evaluation"] = evaluation
        save_risk(self.directory, self.snapshot, self.config, metadata)
        cutoff = evaluation["safe_cutoff"]
        if self.retain_constraint_pair and cutoff["status"] == "ready":
            save_risk(self.directory / "risk_constraint", self.snapshot, self.config, metadata)
            self.constraint_snapshot, self.constraint_cutoff = self.snapshot, cutoff
        self.evaluation, self.validation_plot_data = evaluation, samples
        self.emit(
            "risk_cutoff_updated", generation=int(self.snapshot.generation),
            mode=self.config.cutoff_mode, cutoff=cutoff["threshold"], status=cutoff["status"],
        )

    def observe(self, trajectory, path):
        self.replay.add(trajectory, path, self.randomness.host)

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("Risk-function training failed") from error

    def start(self):
        self.wait()
        if not self.config.enabled:
            return False
        plan = self.replay.prepare(
            self.randomness.host, None if self.snapshot is None else self.snapshot.normalization
        )
        if plan is None:
            self.emit(
                "risk_not_ready",
                reason="both completed training outcomes are required",
                state_counts=self.replay.counts(),
            )
            return False
        self.generation += 1

        def work():
            started = time.perf_counter()
            try:
                from sit_lmpc.data.checkpoints import save_risk

                snapshot, losses = fit_risk(self.state, plan, self.config, self.generation)
                dtype = snapshot.normalization.low.dtype
                evaluation, validation_plot_data = evaluate(
                    snapshot,
                    plan.validation,
                    self.config.batch_size,
                    self.config,
                    self.seed,
                    return_samples=True,
                )
                cutoff = evaluation["safe_cutoff"]
                extra_sizes = (
                    self.diagnostic_batch_size(cutoff)
                    if callable(self.diagnostic_batch_size) else self.diagnostic_batch_size
                )
                extra_counts = (
                    extra_sizes if isinstance(extra_sizes, tuple) else (extra_sizes,)
                )
                for count in dict.fromkeys((1, self.config.batch_size, *extra_counts)):
                    if count is None:
                        continue
                    predict_risk(
                        inference_snapshot(snapshot),
                        jnp.zeros((count, 5), dtype=dtype),
                        jnp.zeros((count, 2), dtype=dtype)
                        if self.config.input_mode == "state_action"
                        else None,
                    ).block_until_ready()
                metadata = {**plan.metadata, "context": self.context, "evaluation": evaluation}
                save_risk(self.directory, snapshot, self.config, metadata)
                if self.retain_constraint_pair and cutoff["status"] == "ready":
                    if cutoff.get("generation") != snapshot.generation:
                        raise ValueError("risk cutoff generation differs from risk model")
                    save_risk(self.directory / "risk_constraint", snapshot, self.config, metadata)
                    self.constraint_snapshot = snapshot
                    self.constraint_cutoff = cutoff
                self.state, self.evaluation, self.validation_plot_data, self.snapshot = (
                    snapshot.state,
                    evaluation,
                    validation_plot_data,
                    snapshot,
                )
                self.emit(
                    "risk_training",
                    generation=self.generation,
                    loss=losses[-1],
                    gradient_steps=int(snapshot.state.step),
                    input_mode=self.config.input_mode,
                    seconds=time.perf_counter() - started,
                    evaluation=evaluation,
                    sampling=plan.metadata,
                )
            except Exception as error:
                self.error = error

        self.thread = threading.Thread(target=work, name="risk-training")
        self.thread.start()
        return True

    def save_report(self):
        report = {
            "generation": self.generation,
            "status": "trained" if self.snapshot is not None else "both_training_outcomes_missing",
            **input_metadata(self.config.input_mode),
            "state_counts": self.replay.counts(),
            "evaluation": self.evaluation,
            "safe_cutoff": None if self.evaluation is None else self.evaluation.get("safe_cutoff"),
        }
        (self.directory / "risk_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
        if self.plots and self.evaluation is not None and self.evaluation["states"] is not None:
            figure = None
            try:
                import matplotlib.pyplot as plt

                figure, axis = plt.subplots(figsize=(5, 4))
                rows = self.evaluation["states"]["reliability"]
                axis.plot([0, 1], [0, 1], "--", color="gray")
                axis.plot([r["predicted"] for r in rows], [r["observed"] for r in rows], "o-")
                cutoff = self.evaluation.get("safe_cutoff") or {}
                if cutoff.get("status") == "ready":
                    axis.axvline(
                        cutoff["threshold"], color="tab:orange", linestyle=":", label="risk cutoff"
                    )
                    axis.legend()
                axis.set(
                    xlabel="predicted failure probability",
                    ylabel="observed failure fraction",
                    title="Held-out risk reliability",
                    xlim=(0, 1),
                    ylim=(0, 1),
                )
                figure.tight_layout()
                figure.savefig(self.directory / "risk_calibration.png")
            except Exception as error:
                self.emit("risk_plot_error", message=str(error))
            finally:
                if figure is not None:
                    plt.close(figure)
