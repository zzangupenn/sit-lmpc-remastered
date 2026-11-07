from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from sit_lmpc.config import load_config
from sit_lmpc.config.schema import ValueConfig
from sit_lmpc.data.checkpoints import load_value, save_value
from sit_lmpc.training.features import inference_snapshot
from sit_lmpc.training.value import initial_state, fit_value, predict_value, train_step, ValueTrainer


def settings(mode='state'):
    return ValueConfig(model='mlp', input_mode=mode, hidden_dims=16, layers=2,
                       activation='swish', batch_size=16, epochs=30, learning_rate=.01,
                       boundary_samples=0, samples=4)


def training_data(mode):
    rng = np.random.default_rng(42)
    x = rng.uniform(size=(48, 7 if mode == 'state_action' else 5))
    return np.column_stack((x, -20 + 20 * x[:, 0]))


@pytest.mark.parametrize('mode', ['state', 'state_action'])
def test_mlp_learns_and_restores_deterministic_state_or_action_value(tmp_path, mode):
    config = settings(mode)
    assert load_config(overrides={'value': {'model': 'mlp'}}).value.model == 'mlp'
    data = training_data(mode)
    snapshot, losses = fit_value(initial_state(config, jax.random.key(7)), data, config, 4., 5, 2)
    assert snapshot.model_type == 'mlp'
    assert losses[-1] < losses[0]
    assert losses[-1] < .02
    states = jnp.column_stack((data[:8, :5], np.zeros((8, 2))))
    actions = jnp.asarray(data[:8, 5:7]) if mode == 'state_action' else None
    a = predict_value(snapshot, states, jax.random.key(1), 1, actions)
    b = predict_value(snapshot, states, jax.random.key(100), 20, actions)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, predict_value(inference_snapshot(snapshot), states,
                                                  jax.random.key(5), 1, actions))
    assert np.isfinite(a).all()
    save_value(tmp_path, snapshot, config)
    assert json.loads((tmp_path / 'value_metadata.json').read_text())['model'] == 'mlp'
    restored = load_value(tmp_path, config)
    assert restored.model_type == 'mlp'
    np.testing.assert_array_equal(a, predict_value(restored, states, jax.random.key(2), 3, actions))
    with pytest.raises(ValueError, match='differs: model'):
        load_value(tmp_path, replace(config, model='normalizing_flow'))
    with pytest.raises(ValueError, match='differs: layers'):
        load_value(tmp_path, replace(config, layers=3))


def test_mlp_masked_mse_ignores_padded_samples():
    state = TrainState.create(apply_fn=lambda p, x: x[:, :1] * p['weight'],
                              params={'weight': jnp.asarray(0.)}, tx=optax.sgd(.01))
    _, loss = train_step(state, jnp.ones((3, 5)), jnp.array([[1.], [2.], [1000.]]),
                         jnp.array([True, True, False]), 'mlp')
    assert float(loss) == pytest.approx(2.5)


def test_background_mlp_trainer_publishes_compatible_snapshot(tmp_path):
    config = replace(settings(), epochs=2)
    trainer = ValueTrainer(config, 4., 10, tmp_path, prediction_batch_size=8)
    trainer.start(training_data('state'))
    trainer.wait()
    assert trainer.snapshot.model_type == 'mlp'
    assert trainer.snapshot.generation == 1
    assert json.loads((tmp_path / 'value_metadata.json').read_text())['value_config']['model'] == 'mlp'
