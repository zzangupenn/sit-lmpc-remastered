"""Optional HalfCheetah-v5 physics; import only in the isolated RL environment."""

import hashlib
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from dyna_dist.domains import get_domain


class FrictionHalfCheetah(gym.Wrapper):
    def __init__(self, friction, *, max_episode_steps=1000, xml_file=None):
        if not np.isfinite(friction) or friction <= 0:
            raise ValueError("friction must be positive and finite")
        kwargs = {} if xml_file is None else {"xml_file": str(xml_file)}
        env = gym.make("HalfCheetah-v5", max_episode_steps=max_episode_steps, **kwargs)
        super().__init__(env)
        raw = self.unwrapped
        if (raw.model.nq, raw.model.nv, raw.model.nu) != (9, 9, 6):
            self.close()
            raise ValueError("expected standard HalfCheetah nq=9,nv=9,nu=6")
        self.friction = float(friction)
        self.original_friction = raw.model.geom_friction.copy()
        # The standard model disables robot self-collision. Set both sides of every
        # ground contact; floor-only edits can be dominated by the other geom.
        raw.model.geom_friction[:, 0] = self.friction
        self.integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.integration_size = mujoco.mj_stateSize(raw.model, self.integration_spec)
        self.max_episode_steps = max_episode_steps

    def physical_state(self):
        return np.concatenate((self.unwrapped.data.qpos, self.unwrapped.data.qvel)).copy()

    def integration_state(self):
        result = np.empty(self.integration_size)
        mujoco.mj_getState(self.unwrapped.model, self.unwrapped.data, result, self.integration_spec)
        return result

    def restore(self, integration_state):
        mujoco.mj_setState(
            self.unwrapped.model,
            self.unwrapped.data,
            np.asarray(integration_state),
            self.integration_spec,
        )
        warmstart = self.unwrapped.data.qacc_warmstart.copy()
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)
        self.unwrapped.data.qacc_warmstart[:] = warmstart
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0
        return self.unwrapped._get_obs()

    def contact_frictions(self):
        raw = self.unwrapped
        floor = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        return [
            contact.friction.copy().tolist()
            for contact in raw.data.contact
            if floor in (contact.geom1, contact.geom2)
        ]

    def physics_metadata(self):
        raw = self.unwrapped
        path = Path(raw.fullpath)
        return {
            "environment": "HalfCheetah-v5",
            "friction": self.friction,
            "domain": get_domain("halfcheetah").metadata(),
            "timestep": float(raw.dt),
            "integrator_timestep": float(raw.model.opt.timestep),
            "frame_skip": raw.frame_skip,
            "max_episode_steps": self.max_episode_steps,
            "xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "geom_friction": raw.model.geom_friction.tolist(),
            "original_geom_friction": self.original_friction.tolist(),
            "body_mass": raw.model.body_mass.tolist(),
            "body_inertia": raw.model.body_inertia.tolist(),
            "dof_damping": raw.model.dof_damping.tolist(),
            "actuator_gear": raw.model.actuator_gear.tolist(),
            "action_low": self.action_space.low.tolist(),
            "action_high": self.action_space.high.tolist(),
            "observation": "qpos[1:] followed by qvel; identity preprocessing",
            "integration_state_spec": int(self.integration_spec),
        }

    def xml_bytes(self):
        return Path(self.unwrapped.fullpath).read_bytes()


def check_friction(seed=71, horizon=100):
    """Matched controls/initial integration state plus actual contact coefficients."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.8, 0.8, (horizon, 6))
    results, initial = {}, None
    for friction in (0.2, 0.8):
        env = FrictionHalfCheetah(friction)
        try:
            env.reset(seed=seed)
            if initial is None:
                initial = env.integration_state()
            else:
                env.restore(initial)
            paths, contacts = [], []
            for action in actions:
                env.step(action)
                paths.append(env.physical_state())
                contacts += env.contact_frictions()
            if not contacts or not all(
                np.isclose(c[0], friction) and np.isclose(c[1], friction) for c in contacts
            ):
                raise AssertionError(
                    "effective ground-contact friction differs from requested value"
                )
            results[friction] = np.asarray(paths)
        finally:
            env.close()
    difference = float(np.sqrt(np.mean((results[0.2] - results[0.8]) ** 2)))
    if difference < 1e-6:
        raise AssertionError("matched action probes show no friction sensitivity")
    return {
        "seed": seed,
        "horizon": horizon,
        "state_rmse_between_frictions": difference,
        "effective_contact_frictions_verified": [0.2, 0.8],
    }
