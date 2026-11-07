"""Pure gradient kernels and a single-writer background value learner."""

from functools import partial
from dataclasses import dataclass, replace
from pathlib import Path
from copy import deepcopy
import threading
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sit_lmpc.models.value.flow import NeuralSplineFlow
from sit_lmpc.models.value.mlp import MLPValue
from sit_lmpc.data.normalization import Normalization
from sit_lmpc.training.features import features, feature_names, inference_snapshot


@struct.dataclass
class ValueSnapshot:
    state: TrainState
    normalization: Normalization
    generation: int
    input_mode: str = struct.field(pytree_node=False, default="state")
    model_type: str = struct.field(pytree_node=False, default="normalizing_flow")


def initial_state(config, key):
    count = len(feature_names(config.input_mode))
    if config.model == "mlp":
        model = MLPValue(config.hidden_dims, config.layers, config.activation)
        params = model.init(key, jnp.zeros((1, count)))
    elif config.model == "normalizing_flow":
        model = NeuralSplineFlow(
            n_dim=1,
            n_context=count,
            hidden_dims=[config.hidden_dims, config.hidden_dims],
            n_transforms=config.layers,
            activation=config.activation,
            n_bins=config.bins,
        )
        params = model.init(key, jnp.zeros((1, 1)), jnp.zeros((1, count)))
    else:
        raise ValueError("value.model must be normalizing_flow or mlp")
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(8), optax.adam(config.learning_rate)),
    )


@partial(jax.jit, static_argnames=("model_type",))
def train_step(state, inputs, targets, mask, model_type="normalizing_flow"):
    def loss(params):
        if model_type == "mlp":
            prediction = state.apply_fn(params, inputs)
            terms = jnp.mean((prediction - targets) ** 2, axis=-1)
        else:
            terms = -state.apply_fn(params, targets, inputs)
        return jnp.sum(terms * mask) / jnp.maximum(mask.sum(), 1)

    value, gradients = jax.value_and_grad(loss)(state.params)
    return state.apply_gradients(grads=gradients), value


@partial(jax.jit, static_argnames=("model_type",))
def train_epoch(state, batches_x, batches_y, masks, model_type="normalizing_flow"):
    def update(state, batch):
        return train_step(state, *batch, model_type)

    state, losses = jax.lax.scan(update, state, (batches_x, batches_y, masks))
    counts = masks.sum(axis=1)
    return state, jnp.sum(losses * counts) / jnp.maximum(counts.sum(), 1)


@partial(jax.jit, static_argnames=("samples",))
def predict_value(snapshot, states, key, samples=20, actions=None):
    inputs = snapshot.normalization.encode_inputs(features(states, snapshot.input_mode, actions))
    if snapshot.model_type == "mlp":
        values = snapshot.state.apply_fn(snapshot.state.params, inputs)
    else:
        latent = jax.random.normal(key, (samples * len(inputs), 1)) / 10
        context = jnp.tile(inputs, (samples, 1))
        # The retained NSF exposes sampling from supplied base values.
        values = snapshot.state.apply_fn(
            snapshot.state.params, latent, context, method=NeuralSplineFlow.sample
        )
        values = values.reshape(samples, len(inputs), 1).mean(axis=0)
    values = snapshot.normalization.decode_targets(values)
    return jnp.clip(values, snapshot.normalization.low[-1], snapshot.normalization.high[-1])[:, 0]


@dataclass(frozen=True)
class TrainingRandomness:
    boundary: np.ndarray
    permutations: tuple[np.ndarray, ...]

    @classmethod
    def allocate(cls, rng, count, config):
        # Allocate on the caller thread, not according to worker scheduling.
        boundary = rng.uniform(
            0, 1, (config.boundary_samples, len(feature_names(config.input_mode)))
        )
        permutations = tuple(
            rng.permutation(count + config.boundary_samples) for _ in range(config.epochs)
        )
        return cls(boundary, permutations)


def fit_value(state, data, config, half_width, seed, generation, randomness=None):
    data = np.asarray(data).copy()
    count = len(feature_names(config.input_mode))
    if (
        data.ndim != 2
        or data.shape[1] != count + 1
        or len(data) == 0
        or not np.isfinite(data).all()
    ):
        raise ValueError(f"value training requires finite (N, {count + 1}) features/target data")
    if randomness is None:
        randomness = TrainingRandomness.allocate(np.random.default_rng(seed), len(data), config)
    normalization = Normalization.fit(data)
    low, high = np.asarray(normalization.low), np.asarray(normalization.high)
    boundary = low[:-1] + randomness.boundary * (high[:-1] - low[:-1])
    boundary[:, 1] += np.where(boundary[:, 1] <= 0, -half_width, half_width)
    targets = np.concatenate([data[:, -1:], np.full((len(boundary), 1), low[-1])])
    inputs = np.concatenate([data[:, :count], boundary])
    x = normalization.encode_inputs(jnp.asarray(inputs))
    y = normalization.encode_targets(jnp.asarray(targets))
    batch = min(config.batch_size, len(inputs))
    total = ((len(inputs) + batch - 1) // batch) * batch
    mask = jnp.asarray(np.arange(total) < len(inputs)).reshape(-1, batch)
    losses = []
    for permutation in randomness.permutations:
        indices = np.pad(permutation, (0, total - len(inputs)), mode="wrap")
        state, loss = train_epoch(
            state, x[indices].reshape(-1, batch, count), y[indices].reshape(-1, batch, 1), mask,
            config.model,
        )
        losses.append(float(loss))
    if not np.isfinite(losses).all() or not all(
        np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state.params)
    ):
        raise FloatingPointError("nonfinite value training loss or parameters")
    return ValueSnapshot(state, normalization, generation, config.input_mode, config.model), losses


class ValueTrainer:
    def __init__(
        self,
        config,
        half_width,
        seed,
        directory,
        emit=lambda *args, **kwargs: None,
        randomness=None,
        prediction_batch_size=None,
        context=None,
    ):
        self.config, self.half_width, self.seed = config, half_width, seed
        self.directory, self.emit = Path(directory), emit
        from sit_lmpc.experiments.randomness import ExperimentRandomness

        self.randomness = randomness if randomness is not None else ExperimentRandomness(seed)
        self.state = initial_state(config, self.randomness.next_key())
        self.snapshot = None
        self.thread = None
        self.error = None
        self.generation = 0
        self.prediction_batch_size, self.context = prediction_batch_size, context

    def warmup(self, snapshot):
        if self.prediction_batch_size is not None:
            counts = self.prediction_batch_size
            if isinstance(counts, int):
                counts = (counts,)
            for count in dict.fromkeys(counts):
                self._warmup_prediction(snapshot, count)

    def _warmup_prediction(self, snapshot, count):
        predict_value(
            inference_snapshot(snapshot),
            jnp.zeros((count, 7), dtype=snapshot.normalization.low.dtype),
            jax.random.key(self.seed),
            self.config.samples,
            jnp.zeros((count, 2), dtype=snapshot.normalization.low.dtype)
            if snapshot.input_mode == "state_action"
            else None,
        ).block_until_ready()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("Value-function training failed") from error

    def start(self, data, epochs=None):
        self.wait()
        # Snapshot the dataset before the worker runs; inference sees only a
        # completed immutable weights/normalization pair.
        data = np.asarray(data).copy()
        count = len(feature_names(self.config.input_mode))
        if (
            data.ndim != 2
            or data.shape[1] != count + 1
            or len(data) == 0
            or not np.isfinite(data).all()
        ):
            raise ValueError(f"value training requires finite (N, {count + 1}) data")
        fit_config = self.config if epochs is None else replace(self.config, epochs=epochs)
        randomness = TrainingRandomness.allocate(self.randomness.host, len(data), fit_config)
        self.generation += 1
        context = deepcopy(self.context)

        def work():
            try:
                from sit_lmpc.data.checkpoints import save_value

                snapshot, losses = fit_value(
                    self.state,
                    data,
                    fit_config,
                    self.half_width,
                    self.seed,
                    self.generation,
                    randomness,
                )
                self.warmup(snapshot)
                if context is None:
                    save_value(self.directory, snapshot, self.config)
                else:
                    save_value(self.directory, snapshot, self.config, context)
                self.state = snapshot.state
                self.snapshot = snapshot
                self.emit(
                    "training",
                    generation=self.generation,
                    epochs=len(losses),
                    loss=losses[-1],
                    gradient_steps=int(snapshot.state.step),
                )
            except Exception as error:
                self.error = error

        self.thread = threading.Thread(target=work, name="value-training")
        self.thread.start()

    def resume(self, directory):
        from sit_lmpc.data.checkpoints import load_value

        self.wait()
        snapshot = load_value(directory, self.config, self.seed, template_state=self.state)
        self.snapshot, self.state, self.generation = (
            snapshot,
            snapshot.state,
            int(snapshot.generation),
        )
