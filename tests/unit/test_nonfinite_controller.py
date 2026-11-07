from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from sit_lmpc.config.schema import ExperimentConfig
from sit_lmpc.planners.common import returns, weights, to_frenet
from sit_lmpc.planners.lmppi import (
    debug_reward_terms, finish_candidates, propose, select_candidate, select_risk_candidate,
    terminal_summary,
)


@pytest.mark.parametrize("problem", ["state", "value", "density", "noise"])
@pytest.mark.parametrize("fixed", [False, True])
def test_invalid_sample_has_zero_weight_and_matches_removing_it(problem, fixed):
    config = replace(ExperimentConfig().planner, samples=5, horizon=3)
    rng = np.random.default_rng(7)
    previous = jnp.zeros((3, 2))
    noise = jnp.asarray(rng.normal(0, .1, (5, 3, 2)))
    states = jnp.zeros((5, 3, 7)).at[:, :, 1].set(jnp.asarray(rng.normal(0, .2, (5, 3))))
    values = jnp.asarray(rng.normal(size=(5, 3)))
    distances = jnp.asarray([.2, .4, .1, .5, .8])
    lambdas = jnp.asarray([[0., 0.], [2., 0.], [2., 2.]])
    keep = np.array([0, 2, 3, 4])
    if problem == "state":
        states = states.at[1, 0, 6].set(jnp.nan)
    elif problem == "value":
        values = values.at[1, 0].set(jnp.nan)
    elif problem == "density":
        distances = distances.at[1].set(jnp.inf)
    else:
        noise = noise.at[1, 0, 0].set(jnp.nan)
    terms = debug_reward_terms(previous, noise, states, values, distances, lambdas, config,
                               fixed_density_cost=fixed)
    np.testing.assert_array_equal(terms["valid"], [True, False, True, True, True])
    w = weights(returns(terms["reward"]), config.temperature, config.damping,
                valid=terms["valid"][None, :, None])
    assert np.isfinite(w).all()
    np.testing.assert_array_equal(w[:, 1], np.zeros((3, 3)))
    actual = propose(previous, noise, states, values, distances, lambdas, config,
                     fixed_density_cost=fixed)
    expected = propose(previous, noise[keep], states[keep], values[keep], distances[keep], lambdas,
                       replace(config, samples=4), fixed_density_cost=fixed)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_all_invalid_samples_have_no_weight_or_control_noise_contribution():
    config = replace(ExperimentConfig().planner, samples=4, horizon=2)
    previous = jnp.full((2, 2), .2)
    noise = jnp.full((4, 2, 2), jnp.nan)
    states, values = jnp.zeros((4, 2, 7)), jnp.zeros((4, 2))
    distances, lambdas = jnp.full(4, jnp.inf), jnp.asarray([[0., 0.], [1., 1.]])
    terms = debug_reward_terms(previous, noise, states, values, distances, lambdas, config)
    w = weights(returns(terms["reward"]), config.temperature, config.damping,
                valid=terms["valid"][None, :, None])
    np.testing.assert_array_equal(w, np.zeros((2, 4, 2)))
    result = propose(previous, noise, states, values, distances, lambdas, config)
    np.testing.assert_array_equal(result, np.broadcast_to(previous, (2, 2, 2)))


def test_nonfinite_candidates_are_excluded_from_plain_and_risk_selection():
    states = jnp.zeros((4, 2, 7)).at[0, 0, 6].set(jnp.nan)
    distances = jnp.asarray([0., .1, .1, jnp.nan])
    values = jnp.asarray([0., 1., 100., 0.])
    assert int(select_candidate(states, distances, 4., .03, values)[0]) == 1
    risk = jnp.zeros((4, 2))
    selected = select_risk_candidate(states, distances, risk, .5, "terminal", 4., .03, values)
    assert int(selected[0]) == 1 and int(selected[4]) == 2


def test_candidate_finish_counts_invalid_actions_and_cartesian_paths():
    config = replace(ExperimentConfig().planner, samples=3, horizon=2)
    states = jnp.zeros((3, 2, 7))
    actions = jnp.zeros((3, 2, 2)).at[0, 0, 0].set(jnp.nan)
    cartesian = states.at[1, 0, 6].set(jnp.nan)
    packet = finish_candidates(states, jnp.zeros(3), None, None, None, None, actions,
                               cartesian, None, None, jnp.zeros((3, 2)), config)
    assert int(packet["index"]) == 2 and int(packet["finite_candidate_count"]) == 1
    assert np.isfinite(packet["actions"]).all() and np.isfinite(packet["trajectory"]).all()


def test_terminal_window_uses_only_finite_complete_rollouts():
    states = jnp.zeros((3, 2, 7)).at[:, -1, 0].set(jnp.asarray([1., 3., 100.]))
    states = states.at[2, 0, 6].set(jnp.inf)
    lo, hi, available = terminal_summary(states)
    assert bool(available) and float(lo) == 1. and float(hi) == 3.
    assert not bool(terminal_summary(states.at[:, 0, 6].set(jnp.nan))[2])


def test_nonfinite_progress_cannot_change_the_frame_for_other_rollouts():
    class Track:
        s_frame_max = 100.

        def vmap_cartesian_to_frenet_jax_jit(self, poses, indices):
            return poses

    track = Track()
    states = jnp.zeros((3, 2, 7)).at[:, :, 0].set(jnp.asarray([[98., 99.], [1., 2.], [jnp.nan, jnp.nan]]))
    actual = to_frenet(track, states, jnp.arange(3), 99.)
    reference = to_frenet(track, states[:2], jnp.arange(3), 99.)
    np.testing.assert_array_equal(actual[:2], reference)
    np.testing.assert_array_equal(actual[1, :, 0], [101., 102.])


def test_all_invalid_controller_samples_stop_before_overwriting_warmstart(monkeypatch):
    from types import SimpleNamespace
    from sit_lmpc.experiments.runner import build_vehicle_dynamics
    from sit_lmpc.planners import lmppi
    from sit_lmpc.track import Track, map_path
    from density_fixture import score_snapshot

    base = ExperimentConfig()
    config = replace(base.planner, safe_set_query="nf_density", samples=4, horizon=2,
                     lambda_count=2, hull_precompile=False,
                     safe_set_density=replace(base.planner.safe_set_density,
                                              penalty_normalization="sample_max"))
    _, model = build_vehicle_dynamics(base)
    track = Track.from_track_path(map_path("custom1"))
    controller = lmppi.LMPPI(config, model, track)
    controller.density_snapshot = score_snapshot(0., 1.)
    controller.actions = jnp.full((2, 2), .1, dtype=model.params.dtype)
    initial = np.asarray(controller.actions).copy()
    monkeypatch.setattr(lmppi, "sample_rollouts", lambda *args: (
        jnp.full((4, 2, 7), jnp.nan), jnp.zeros((4, 2, 2)), jnp.zeros((4, 2, 2))))
    monkeypatch.setattr(lmppi, "predict_value", lambda snapshot, states, *args: jnp.zeros(len(states)))
    pose = track.frenet_to_cartesian(0., 0., 0.)
    state = jnp.asarray([pose[0], pose[1], 0., 10., pose[2], 0., 0.], model.params.dtype)
    value = SimpleNamespace(input_mode="state", generation=1)
    with pytest.raises(FloatingPointError, match="0/4 valid samples"):
        controller.update(state, np.array([0., 0., 0., 10., 0., 0., 0.]), value, None)
    np.testing.assert_array_equal(controller.actions, initial)
