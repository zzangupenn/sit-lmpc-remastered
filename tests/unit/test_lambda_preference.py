"""SIT's first-minimum rule applies to the complete active lambda vector."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import PlannerConfig
from sit_lmpc.planners.lmppi import (
    finish_candidates,
    select_candidate,
    select_risk_candidate,
)
from sit_lmpc.planners.probabilistic import lambda_table, select_constrained_candidate


def base_table(config, dtype=jnp.float32):
    levels = np.arange(config.lambda_count) * config.lambda_max / config.lambda_count
    pairs = np.stack(np.meshgrid(levels, levels), axis=-1).reshape(-1, 2)
    return jnp.asarray(pairs[np.argsort(np.linalg.norm(pairs, axis=1))], dtype)


@pytest.mark.parametrize(
    "chance,risk", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("limit", [8, 256])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_all_active_terms_are_sorted_without_changing_shape_or_inactive_axes(
    chance, risk, limit, dtype
):
    config = PlannerConfig()
    config = replace(
        config,
        probabilistic_safety=replace(config.probabilistic_safety, max_candidates=limit),
        risk_constraint=replace(config.risk_constraint, max_candidates=limit),
    )
    original = base_table(config, dtype)
    actual = lambda_table(original, config, chance, risk)
    if not chance and not risk:
        assert actual is original  # Preserve original SIT ordering bit-for-bit.
    expected_count = min(config.lambda_count**2, limit) if chance or risk else len(original)
    assert actual.shape == (expected_count, 2 + 2 * chance + risk)
    assert actual.dtype == dtype
    norms = np.linalg.norm(np.asarray(actual), axis=1)
    assert np.all(np.diff(norms) >= 0)
    np.testing.assert_array_equal(actual[0], 0)
    np.testing.assert_array_equal(actual, lambda_table(original, config, chance, risk))


def test_sort_retains_sampled_vectors_and_original_order_for_equal_norms():
    config = PlannerConfig(lambda_count=2)
    # The original risk-only design cycles through levels 0, 5, 10, 15.
    pairs = jnp.asarray([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    config = replace(config, risk_constraint=replace(config.risk_constraint, lambda_count=4))
    unsorted = np.column_stack((pairs, [0.0, 5.0, 10.0, 15.0]))
    expected = unsorted[np.argsort(np.linalg.norm(unsorted, axis=1), kind="stable")]
    np.testing.assert_array_equal(lambda_table(pairs, config, False, True), expected)
    # Every vector ties in norm: stable sorting must not invent a new tie-break.
    tied = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    config = replace(config, risk_constraint=replace(config.risk_constraint, lambda_count=1))
    np.testing.assert_array_equal(
        lambda_table(tied, config, False, True), np.column_stack((tied, np.zeros(4)))
    )


@pytest.mark.parametrize("risk", [False, True])
@pytest.mark.parametrize("action_value", [False, True])
def test_chance_gate_selects_smallest_complete_vector_not_nominal_table_order(risk, action_value):
    config = PlannerConfig()
    config = replace(
        config, probabilistic_safety=replace(config.probabilistic_safety, lambda_count=10)
    )
    table = np.asarray(lambda_table(base_table(config), config, True, risk))
    count = len(table)
    chance = {
        "available": jnp.ones(count, bool),
        "boundary_upper": jnp.zeros(count),
        "terminal_hull_upper": jnp.zeros(count).at[0].set(1.0),
    }
    # All later candidates pass both probability gates. Smaller raw probability
    # and larger Q must not outrank the smallest-lambda feasible candidate.
    chance["terminal_hull_upper"] = chance["terminal_hull_upper"].at[1].set(0.05)
    risk_values = jnp.full((count, 3), 0.1) if risk else None
    values = jnp.arange(count, dtype=jnp.float32) if action_value else None
    result = select_constrained_candidate(
        jnp.zeros((count, 3, 7)),
        jnp.zeros(count),
        chance,
        risk_values,
        0.5,
        0.05,
        0.05,
        "terminal",
        4.0,
        0.03,
        values,
    )
    assert int(result[0]) == 1
    assert int(result[3]) == count - 1 and not bool(result[4])
    assert np.linalg.norm(table[1]) == pytest.approx(np.linalg.norm(table[1:], axis=1).min())
    # Reproduce the pre-fix chance-only failure with this known sampled vector.
    if not risk:
        np.testing.assert_array_equal(table[1], [3.75, 1.25, 2.0, 2.0])


@pytest.mark.parametrize("selector", ["nominal", "risk", "chance", "joint"])
@pytest.mark.parametrize("fallback", [False, True])
def test_action_values_never_replace_first_equal_geometric_score(selector, fallback):
    states = jnp.zeros((3, 3, 7))
    distances = jnp.asarray([0.1, 0.1, 0.2])
    values = jnp.asarray([-100.0, 100.0, 1000.0])
    probabilities = jnp.full((3, 3), 1.0 if fallback else 0.1)
    if selector == "nominal":
        result = select_candidate(states, distances, 4.0, 0.03, values)
    elif selector == "risk":
        result = select_risk_candidate(
            states, distances, probabilities, 0.5, "terminal", 4.0, 0.03, values
        )
        assert bool(result[5]) == fallback
    else:
        chance = {
            "available": jnp.ones(3, bool),
            "boundary_upper": jnp.full(3, 1.0 if fallback else 0.01),
            "terminal_hull_upper": jnp.zeros(3),
        }
        result = select_constrained_candidate(
            states,
            distances,
            chance,
            probabilities if selector == "joint" else None,
            0.5,
            0.05,
            0.05,
            "terminal",
            4.0,
            0.03,
            values,
        )
        assert bool(result[4]) == fallback
    assert int(result[0]) == 0


def test_constraint_violation_still_outranks_lambda_preference():
    states = jnp.zeros((3, 3, 7))
    values = jnp.asarray([1000.0, 0.0, 100.0])
    # A smaller lambda never excuses a larger geometric violation.
    result = select_candidate(states, jnp.asarray([0.2, 0.1, 0.3]), 4.0, 0.03, values)
    assert int(result[0]) == 1


def test_finish_candidates_keeps_actions_trajectories_and_lambdas_aligned():
    config = PlannerConfig()
    lambdas = lambda_table(base_table(config), config, True, False)
    count = len(lambdas)
    states = jnp.zeros((count, 3, 7))
    labels = jnp.arange(count, dtype=states.dtype)
    actions = jnp.broadcast_to(labels[:, None, None], (count, 3, 2))
    trajectories = jnp.broadcast_to(labels[:, None, None], (count, 3, 7))
    chance = {
        "available": jnp.ones(count, bool),
        "boundary": jnp.zeros(count),
        "terminal_hull": jnp.zeros(count),
        "boundary_upper": jnp.zeros(count),
        "terminal_hull_upper": jnp.zeros(count).at[0].set(1.0),
    }
    result = finish_candidates(
        states,
        jnp.zeros(count),
        chance,
        None,
        None,
        labels,
        actions,
        trajectories,
        None,
        None,
        lambdas,
        config,
    )
    assert int(result["index"]) == 1
    np.testing.assert_array_equal(result["actions"], actions[1])
    np.testing.assert_array_equal(result["trajectory"], trajectories[1])
    np.testing.assert_array_equal(result["lambdas"], lambdas[1])
    assert float(result["terminal_value"]) == 1.0
