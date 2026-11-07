"""F1TENTH reset, stepping, coordinate layout, and resource ownership."""

import numpy as np


def vehicle_parameters(friction, mass_ratio=1.0):
    from f1tenth_gym.envs.f110_env import F110Env

    params = F110Env.fullscale_vehicle_params()
    # Preserve the reference runner's footprint convention independently of
    # the ST equations, which do not use length/width.
    params["length"] = params["width"]
    params["mu"] = friction
    # Match the v2 population: mass varies independently; yaw inertia is fixed.
    params["m"] *= mass_ratio
    return params


class F1TenthSimulator:
    def __init__(self, simulation, track_path, render, parameters):
        import gymnasium as gym

        self.simulation, self.render_enabled = simulation, render
        self.env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={
                "seed": simulation.seed,
                "map": str(track_path),
                "timestep": simulation.timestep,
                "integrator_timestep": simulation.integrator_timestep,
                "control_input": ["accl", "steering_speed"],
                "model": "st",
                "observation_config": {"type": "direct"},
                "params": parameters,
                "map_scale": 1.0,
                "enable_rendering": render,
                "enable_scan": False,
                "compute_frenet": True,
                "max_laps": "inf",
                "steer_delay_buffer_size": 0,
            },
            render_mode="unlimited" if render else None,
        )
        self.initialized = False

    def reset(self, state):
        # This fork advances a zero-input step in env.reset(). Reinstall the
        # requested state afterwards, rather than returning that advanced state.
        state = np.asarray(state, dtype=self.simulation.dtype).copy()
        raw = self.env.unwrapped
        scan_rng = raw.sim.agents[0].scan_rng if self.initialized else None
        observation, info = self.env.reset(
            seed=None if self.initialized else self.simulation.seed,
            options={"states": state[None].copy()},
        )
        raw.sim.reset(state[None].copy(), option="state")
        if scan_rng is not None:
            raw.sim.agents[0].scan_rng = scan_rng
        raw.sim.agent_poses[0] = state[[0, 1, 4]]
        raw.sim.agent_steerings[0] = state[2]
        raw.sim.collisions[:] = 0
        raw._update_state()
        raw.sim_time = 0.0
        raw.lap_counts[:] = 0
        raw.lap_times[:] = 0
        raw.lap_times_finish[:] = 0
        raw.agents_prev_s[:] = None
        observation = raw.observation_type.observe()
        if self.render_enabled:
            from copy import deepcopy

            raw.render_obs = deepcopy(observation)
        info = {
            "lap_times": raw.lap_times.copy(),
            "lap_counts": raw.lap_counts.copy(),
            "sim_time": 0.0,
        }
        self.initialized = True
        return self.unpack(observation), info

    def unpack(self, observation):
        cartesian = np.asarray(
            observation["agent_0"]["std_state"], dtype=self.simulation.dtype
        ).copy()
        frenet = cartesian.copy()
        frenet[[0, 1, 4]] = observation["agent_0"]["frenet_pose"]
        return cartesian, frenet

    def snapshot(self):
        """Capture Gym's lap bookkeeping at a completed-trial boundary."""
        raw = self.env.unwrapped
        return {
            "sim_time": float(raw.sim_time),
            "lap_counts": raw.lap_counts.tolist(),
            "lap_times": raw.lap_times.tolist(),
            "lap_times_finish": raw.lap_times_finish.tolist(),
            "agents_prev_s": [
                None if value is None else float(value) for value in raw.agents_prev_s
            ],
        }

    def restore(self, state, bookkeeping):
        result = self.reset(state)
        raw = self.env.unwrapped
        raw.sim_time = float(bookkeeping["sim_time"])
        for name in ("lap_counts", "lap_times", "lap_times_finish", "agents_prev_s"):
            getattr(raw, name)[:] = bookkeeping[name]
        return result

    def step(self, normalized_action, noise):
        control = np.asarray(normalized_action) * np.asarray(self.simulation.action_scale) + noise
        observation, _, terminated, truncated, info = self.env.step(control[None])
        info = {**info, "collision": bool(self.env.unwrapped.collisions[0])}
        return self.unpack(observation), control, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_enabled:
            self.env.render()

    def capture_render(self):
        """Display state only: observing history never resets or steps Gym."""
        from copy import deepcopy

        return deepcopy(self.env.unwrapped.render_obs) if self.render_enabled else None

    def display_observation(self, observation):
        from copy import deepcopy

        if self.render_enabled and observation is not None:
            self.env.unwrapped.render_obs = deepcopy(observation)

    def observation_for_frame(self, frame):
        """Compatibility for recordings predating saved render observations."""
        from copy import deepcopy

        observation = deepcopy(self.env.unwrapped.render_obs)
        state = np.asarray(frame.get("next_state", frame["state"]))
        frenet = np.asarray(frame.get("next_frenet", frame["frenet"]))
        agent = observation["agent_0"]
        agent.update(std_state=state.copy(), state=state.copy(), frenet_pose=frenet[[0, 1, 4]].copy(),
                     lap_time=frame["time"] + self.simulation.timestep,
                     lap_count=frame["lap"] - 1)
        return observation

    def window_open(self):
        renderer = getattr(self.env.unwrapped, "renderer", None)
        window = getattr(renderer, "window", None)
        return self.render_enabled and (window is None or window.isVisible())

    def bind_keyboard(self, keyboard):
        """Use the existing Gym window when a global listener is unavailable."""
        if not self.render_enabled:
            return
        from PyQt6 import QtCore

        renderer = self.env.unwrapped.renderer
        window = getattr(renderer, "window", None)
        if window is None:
            return

        class Keys(QtCore.QObject):
            def eventFilter(self, obj, event):
                if (event.type() == QtCore.QEvent.Type.KeyRelease and not event.isAutoRepeat()
                        and hasattr(obj, "window") and obj.window() is window):
                    key = {QtCore.Qt.Key.Key_Left: "left", QtCore.Qt.Key.Key_Right: "right",
                           QtCore.Qt.Key.Key_Q: "q"}.get(event.key())
                    if key is None and event.text() in "0123456" and event.text():
                        key = event.text()
                    if key is not None:
                        keyboard.focused_command(key)
                return False

        self._history_keys = Keys(window)
        renderer.app.installEventFilter(self._history_keys)

    def close(self):
        # This F1TENTH fork omits renderer initialization in headless mode.
        # Fix the instance boundary rather than changing the shared library.
        if not hasattr(self.env.unwrapped, "renderer"):
            self.env.unwrapped.renderer = None
        self.env.close()
