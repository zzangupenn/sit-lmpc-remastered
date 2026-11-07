from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config import load_config
from sit_lmpc.adapters.f1tenth import vehicle_parameters
from sit_lmpc.dynamics import STModel, STModelBank, bank_step, bank_rollout, step, rollout
from sit_lmpc.dynamics.st import PARAMETER_FIELDS
from sit_lmpc.diagnostics.model_comparison import ModelComparisons


def model(dtype="float32", friction=0.6):
    sim = replace(load_config().simulation, dtype=dtype)
    return STModel.from_vehicle(vehicle_parameters(friction), sim)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scan", [True, False])
def test_batched_members_match_single_model_and_preserve_parameters(dtype, scan):
    belief = model(dtype)
    original = np.asarray(belief.params).copy()
    bank = STModelBank.from_model(belief, (0.4, 0.6, 0.8))
    initial = jnp.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=dtype)
    actions = jnp.tile(jnp.asarray([0.3, 0.1], dtype=dtype), (10, 1))
    one_step = np.asarray(bank_step(bank, initial, actions[0]))
    horizons = np.asarray(bank_rollout(bank, initial, actions, scan))
    tolerance = 2e-5 if dtype == "float32" else 1e-11
    for i in range(3):
        member = bank.member(bank.params[i])
        np.testing.assert_allclose(
            one_step[i], step(member, initial, actions[0]), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            horizons[i], rollout(member, initial, actions, scan), rtol=tolerance, atol=tolerance
        )
    np.testing.assert_array_equal(belief.params, original)
    np.testing.assert_array_equal(
        bank.params[:, 1:], np.broadcast_to(original[1:], (3, len(original) - 1))
    )
    np.testing.assert_allclose(
        horizons[1], rollout(belief, initial, actions, scan), rtol=tolerance, atol=tolerance
    )
    assert np.max(np.abs(horizons[0] - horizons[2])) > 1e-4
    assert horizons.dtype == np.dtype(dtype)


def test_bank_values_and_partial_windows_do_not_recompile(tmp_path):
    bank_step.clear_cache()
    bank_rollout.clear_cache()
    belief = model()
    initial = jnp.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=jnp.float32)
    actions = jnp.zeros((5, 2), dtype=jnp.float32)
    for frictions in ((0.4, 0.7), (0.3, 0.8)):
        bank = STModelBank.from_model(belief, frictions)
        bank_step(bank, initial, actions[0]).block_until_ready()
        bank_rollout(bank, initial, actions).block_until_ready()
    assert bank_step._cache_size() == bank_rollout._cache_size() == 1
    comparison = ModelComparisons(belief, model(friction=0.5), (0.4, 0.7), 5)
    comparison.warmup(initial)
    count = bank_rollout._cache_size()
    state = np.asarray(initial)
    for i in range(8):
        next_state = np.asarray(step(belief, state, actions[0]))
        comparison.observe(state, np.zeros(2), next_state)
        result = comparison.recorded(i)
        assert result.actual.shape == (min(i + 1, 5) + 1, 7)
        np.testing.assert_allclose(result.belief, result.actual, rtol=2e-5, atol=2e-5)
        state = next_state
    assert bank_rollout._cache_size() == count
    assert len(comparison.states) == 6 and len(comparison.actions) == 5


def test_one_model_bank_reproduces_belief_predictor():
    belief = model()
    bank = STModelBank.from_model(belief, (0.6,))
    state = jnp.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=jnp.float32)
    actions = jnp.zeros((4, 2), dtype=jnp.float32)
    np.testing.assert_allclose(
        bank_step(bank, state, actions[0])[0], step(belief, state, actions[0]), rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        bank_rollout(bank, state, actions)[0], rollout(belief, state, actions), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize(
    "frictions", [(), (0,), (-1,), (True,), (float("nan"),), (float("inf"),), (0.4, 0.4), ("0.4",)]
)
def test_invalid_bank_configuration(frictions):
    with pytest.raises(ValueError, match="planner.model_bank.frictions"):
        load_config(overrides={"planner": {"model_bank": {"frictions": frictions}}})


def test_keyboard_view_toggle_must_be_boolean():
    with pytest.raises(ValueError, match="diagnostics.keyboard_controls"):
        load_config(overrides={"diagnostics": {"keyboard_controls": "true"}})


def test_comparison_does_not_draw_randomness(tmp_path):
    np.random.seed(29)
    before = np.random.get_state()
    belief = model()
    reference = model(friction=0.5)
    comp = ModelComparisons(belief, reference, (0.4, 0.7), 3)
    state = np.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=np.float32)
    comp.warmup(state)
    planned = comp.planned(state, np.zeros((3, 2)), 0)
    assert not list(tmp_path.iterdir())
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert planned.actual.shape == (1, 7) and planned.kind == "planned"
    assert float(belief.params[PARAMETER_FIELDS.index("mu")]) == pytest.approx(0.6)
    assert float(reference.params[PARAMETER_FIELDS.index("mu")]) == pytest.approx(0.5)
