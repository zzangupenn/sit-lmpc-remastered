"""Verify mass variation against the actual Gym transition implementation."""

import numpy as np
import pytest

from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
from sit_lmpc.config import load_config
from sit_lmpc.experiments.runner import build_vehicle_dynamics
from sit_lmpc.track import Track, map_path
from dyna_dist.adapters.gym_st import GymST


def test_gym_actually_uses_sampled_mass_with_fixed_yaw_inertia():
    config = load_config(overrides={
        "simulation": {"friction": .43, "mass_ratio": 1.04, "dtype": "float64",
                       "seed_mode": "fixed", "seed": 52},
        "diagnostics": {"render": False, "keyboard_controls": False},
    })
    actual, _ = build_vehicle_dynamics(config)
    track = Track.from_track_path(map_path("custom1"))
    pose = track.frenet_to_cartesian(0, 0, 0)
    state = np.array([pose[0], pose[1], .04, 15, pose[2], .1, .01])
    simulator = F1TenthSimulator(config.simulation, map_path("custom1"), False, actual)
    try:
        simulator.reset(state)
        parameters = simulator.env.unwrapped.sim.agents[0].params
        assert parameters["m"] == pytest.approx(actual["m"])
        assert parameters["I"] == vehicle_parameters(.43)["I"]
        (result, _), control, _, _, _ = simulator.step(np.array([.2, .3]), np.zeros(2))
        truth = GymST(actual, .1, .02).step(state, control)
        np.testing.assert_allclose(simulator.env.unwrapped.sim.agents[0].state, truth,
                                   atol=1e-10, rtol=1e-10)
        # Gym's direct observation is float32 even when the integrator is float64.
        np.testing.assert_allclose(result, truth, atol=1e-6, rtol=1e-6)
        nominal = GymST(vehicle_parameters(.43), .1, .02).step(state, control)
        assert abs(nominal[5] - result[5]) > 1e-5
    finally:
        simulator.close()
