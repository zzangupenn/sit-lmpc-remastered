"""Analytical density fixture for controller nonfinite-path checks."""
from flax.training.train_state import TrainState
import jax.numpy as jnp
import optax
from sit_lmpc.training.safe_set_density import DensitySnapshot


def score_snapshot(cutoff, scale):
    state = TrainState.create(apply_fn=lambda params, targets, context: -targets[:, 0], params={}, tx=optax.sgd(0.))
    return DensitySnapshot(state, jnp.zeros(4), jnp.ones(4), jnp.asarray(100.),
                           jnp.asarray(cutoff), jnp.asarray(scale), 1, penalty_kind="bounded_linear",
                           relaxation_cutoff=jnp.asarray(cutoff))
