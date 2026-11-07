"""Stage 5.1 uses independent trajectory branches, not averaged friction/state recursion."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import SimulationConfig
from sit_lmpc.adapters.f1tenth import vehicle_parameters
from sit_lmpc.dynamics import STModel, STModelBank, STEnsemble, rollout, step
from sit_lmpc.dynamics.ensemble import (
    weighted_states,
    predictor_step,
    predictor_rollout,
    member_rollouts,
    predictor_context,
)


def predictor(dtype="float64", frictions=(0.3, 0.6, 1.0), weights=(0.2, 0.3, 0.5)):
    simulation = SimulationConfig(
        dtype=dtype, timestep=0.1, integrator_timestep=0.02, action_scale=(0.4, 3.5)
    )
    model = STModel.from_vehicle(vehicle_parameters(0.7), simulation)
    return STEnsemble(STModelBank.from_model(model, frictions), jnp.asarray(weights, dtype=dtype))


def reference_mean(members, weights):
    weights = np.asarray(weights)
    mean = np.einsum("m,m...d->...d", weights, members)
    anchor = members[np.argmax(weights)]
    for component in (2, 4, 6):
        angle = np.arctan2(
            np.einsum("m,m...->...", weights, np.sin(members[..., component])),
            np.einsum("m,m...->...", weights, np.cos(members[..., component])),
        )
        mean[..., component] = anchor[..., component] + np.arctan2(
            np.sin(angle - anchor[..., component]), np.cos(angle - anchor[..., component])
        )
    return mean


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scan", [True, False])
def test_mean_of_independent_full_rollouts_matches_direct_states(dtype, scan):
    belief = predictor(dtype)
    state = jnp.asarray([0, 0, 0.18, 13, 3.2, 0.03, 0.05], dtype=dtype)
    actions = jnp.tile(jnp.asarray([0.15, 0.1], dtype=dtype), (10, 1))
    members = np.asarray(member_rollouts(belief, state, actions, scan))
    expected_members = np.asarray(
        [rollout(belief.bank.member(params), state, actions, scan) for params in belief.params]
    )
    tolerance = 1e-5 if dtype == "float32" else 1e-11
    np.testing.assert_allclose(members, expected_members, atol=tolerance, rtol=tolerance)
    mean = np.asarray(predictor_rollout(belief, state, actions, scan))
    np.testing.assert_allclose(
        mean, reference_mean(members, belief.weights), atol=tolerance, rtol=tolerance
    )
    assert mean.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        predictor_step(belief, state, actions[0]), mean[0], atol=tolerance, rtol=tolerance
    )

    # Neither an averaged coefficient nor recursive averaged-state stepping is
    # the expected trajectory under one fixed unknown model for the horizon.
    mean_mu = float(np.asarray(belief.weights) @ np.asarray(belief.params[:, 0]))
    averaged = belief.bank.member(belief.params[0].at[0].set(mean_mu))
    assert np.max(np.abs(mean - np.asarray(rollout(averaged, state, actions, scan)))) > 1e-4
    recursive, current = [], state
    for action in actions:
        current = predictor_step(belief, current, action)
        recursive.append(np.asarray(current))
    assert np.max(np.abs(mean - np.asarray(recursive))) > 1e-4


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_circular_wrap_zero_mass_and_antipodal_ambiguity(dtype):
    states = np.zeros((3, 2, 7), dtype=dtype)
    states[0, :, [2, 4, 6]] = np.pi - 0.1
    states[1, :, [2, 4, 6]] = -np.pi + 0.1
    states[2] = np.nan
    mean = np.asarray(weighted_states(jnp.asarray(states), jnp.asarray([0.5, 0.5, 0], dtype=dtype)))
    assert np.isfinite(mean).all()
    np.testing.assert_allclose(np.cos(mean[:, [2, 4, 6]]), -1, atol=1e-6)
    states = np.zeros((2, 7), dtype=dtype)
    states[1, [2, 4, 6]] = np.pi
    ambiguous = np.asarray(
        weighted_states(jnp.asarray(states), jnp.asarray([0.5, 0.5], dtype=dtype))
    )
    np.testing.assert_array_equal(ambiguous, states[0])


@pytest.mark.parametrize("weights,frictions", [((1,), (0.6,)), ((0, 1, 0), (0.3, 0.6, 1.0))])
def test_one_model_and_one_hot_reproduce_member_exactly(weights, frictions):
    belief = predictor(frictions=frictions, weights=weights)
    state = jnp.asarray([0, 0, 0.1, 12, 7, 0, 0], dtype=jnp.float64)
    actions = jnp.zeros((4, 2))
    member = belief.bank.member(belief.params[np.argmax(weights)])
    np.testing.assert_array_equal(
        predictor_rollout(belief, state, actions), rollout(member, state, actions)
    )
    np.testing.assert_array_equal(
        predictor_step(belief, state, actions[0]), step(member, state, actions[0])
    )


def test_probability_updates_are_dynamic_and_do_not_change_model_support():
    predictor_rollout.clear_cache()
    belief = predictor()
    state = jnp.asarray([0, 0, 0.1, 12, 0, 0, 0], dtype=jnp.float64)
    actions = jnp.zeros((4, 2))
    first = np.asarray(predictor_rollout(belief, state, actions))
    changed = belief.replace(weights=jnp.asarray([0.8, 0.1, 0.1]))
    other = np.asarray(predictor_rollout(changed, state, actions))
    assert np.max(np.abs(first - other)) > 1e-4
    np.testing.assert_array_equal(belief.params, changed.params)
    assert predictor_rollout._cache_size() == 1
    context = predictor_context(changed)
    assert context["kind"] == "ensemble" and "friction" not in context
