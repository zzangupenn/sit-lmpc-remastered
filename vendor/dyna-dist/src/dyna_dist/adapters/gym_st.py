"""Training labels from Gym's ST physics, not our learned/JAX predictor."""

from pathlib import Path
import hashlib
import numpy as np
from dyna_dist.models.profiles import PARAMETER_FIELDS


class GymST:
    def __init__(self, parameters, timestep, integrator_timestep):
        try:
            from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st
            from f1tenth_gym.envs.integrator import RK4Integrator
        except ImportError as error:
            raise ImportError("Collection requires: uv sync --extra collection") from error
        self.parameters = dict(parameters)
        self.params = np.array([parameters[k] for k in PARAMETER_FIELDS], dtype=np.float64)
        self.rhs = vehicle_dynamics_st
        self.integrator = RK4Integrator(timestep, integrator_timestep)

    def step(self, state, control):
        result = self.integrator.integrate(self.rhs, state, control, self.params)
        result[4] = (result[4] + np.pi) % (2 * np.pi) - np.pi
        return result

    @property
    def source_fingerprint(self):
        import inspect

        paths = [
            Path(inspect.getfile(self.rhs.py_func)),
            Path(inspect.getfile(type(self.integrator))),
        ]
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class GymEnvironment:
    """Optional real environment validation, with reset's hidden step undone."""

    def __init__(self, parameters, config, seed):
        import gymnasium as gym

        map_prefix = Path(config.map).expanduser()
        if map_prefix.name.endswith("_map.yaml"):
            map_prefix = map_prefix.with_name(map_prefix.name[:-9])
        if not Path(f"{map_prefix}_map.yaml").is_file():
            raise ValueError(
                "gym_env requires an explicit local map prefix with _map.yaml; "
                "named-map downloading is not used by this adapter"
            )
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "seed": seed,
                "map": str(map_prefix.resolve()),
                "params": parameters,
                "model": "st",
                "timestep": config.timestep,
                "integrator_timestep": config.integrator_timestep,
                "integrator": "rk4",
                "control_input": ["accl", "steering_speed"],
                "observation_config": {"type": "direct"},
                "enable_rendering": False,
                "enable_scan": False,
                "compute_frenet": True,
                "max_laps": "inf",
                "steer_delay_buffer_size": 0,
            },
        )
        self.seed = seed

    def reset(self, state):
        raw = self.env.unwrapped
        # Put the local state on the track, retaining the sampled dynamic components.
        center = raw.track.centerline
        state = state.copy()
        state[0:2] = [center.xs[0], center.ys[0]]
        state[4] = float(center.yaws[0])
        self.env.reset(seed=self.seed, options={"states": state[None].copy()})
        raw.sim.reset(state[None].copy(), option="state")
        raw.sim.agent_poses[0] = state[[0, 1, 4]]
        raw.sim.agent_steerings[0] = state[2]
        raw.sim.collisions[:] = 0
        raw._update_state()
        raw.sim_time = 0.0
        raw.lap_counts[:] = 0
        raw.lap_times[:] = 0
        raw.lap_times_finish[:] = 0
        raw.agents_prev_s[:] = None
        return raw.sim.agents[0].state.copy()

    def step(self, state, control):
        _, _, terminated, truncated, _ = self.env.step(control[None])
        raw = self.env.unwrapped
        return (
            raw.sim.agents[0].state.copy(),
            bool(terminated or truncated),
            bool(raw.collisions[0]),
        )

    def close(self):
        if not hasattr(self.env.unwrapped, "renderer"):
            self.env.unwrapped.renderer = None
        self.env.close()
