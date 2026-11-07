from dataclasses import replace
import threading
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import ValueConfig
from sit_lmpc.data.checkpoints import save_value, load_value
from sit_lmpc.training.value import initial_state, fit_value, predict_value, ValueTrainer


def small_config():
    return ValueConfig(
        hidden_dims=8, layers=2, bins=4, batch_size=16, epochs=3, boundary_samples=10, samples=4
    )


def dataset():
    rng = np.random.default_rng(0)
    data = rng.uniform(size=(33, 6))
    data[:, 0] *= 100
    data[:, 1] = data[:, 1] * 2 - 1
    data[:, -1] = np.linspace(-20, 0, 33)
    return data


def test_gradient_update_and_checkpoint_roundtrip(tmp_path):
    config = small_config()
    original = initial_state(config, jax.random.key(0))
    snapshot, losses = fit_value(original, dataset(), config, 4, 0, 1)
    assert np.isfinite(losses).all()
    assert int(snapshot.state.step) == 9  # Includes the incomplete final batch.
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(original.params), jax.tree.leaves(snapshot.state.params))
    )
    save_value(tmp_path, snapshot, config)
    restored = load_value(tmp_path, config)
    for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(snapshot)):
        np.testing.assert_array_equal(a, b)
    states = jnp.column_stack([jnp.asarray(dataset()[:5, :5]), jnp.zeros((5, 2))])
    prediction = predict_value(restored, states, jax.random.key(1), config.samples)
    assert np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="differs"):
        load_value(tmp_path, replace(config, layers=3))


def test_failed_worker_does_not_publish_snapshot(tmp_path, monkeypatch):
    import sit_lmpc.data.checkpoints as checkpoints

    def fail(*args):
        raise OSError("checkpoint failed")

    monkeypatch.setattr(checkpoints, "save_value", fail)
    trainer = ValueTrainer(replace(small_config(), epochs=1), 4, 0, tmp_path)
    trainer.start(dataset())
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.wait()
    assert trainer.snapshot is None
    assert trainer.thread is None


def test_repeated_fits_are_reproducible():
    config = replace(small_config(), epochs=1)
    state = initial_state(config, jax.random.key(0))
    a, _ = fit_value(state, dataset(), config, 4, 123, 1)
    b, _ = fit_value(state, dataset(), config, 4, 123, 1)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(left, right)
