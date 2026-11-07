"""DeepMind walker/walk model with explicit native/MJX and BRO state conventions.

The model/assets come from MuJoCo Playground's DM Control Suite distribution.
No Gym Walker model, velocity clipping, or Gym termination rules are used here.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from flax import struct
import jax
import jax.numpy as jnp
import mujoco
import numpy as np


class NativeWalker:
    dt = 0.025
    frame_skip = 10
    domain = "dm_walker"

    def __init__(
        self, *, friction=0.7, mass_ratio=1.0, reset_mode="random", max_steps=1000, model_path=None
    ):
        if model_path is None:
            from mujoco_playground._src import mjx_env
            from mujoco_playground._src.dm_control_suite import common

            path = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "walker.xml"
            assets = common.get_assets()
        else:
            path, assets = Path(model_path), None
        xml = path.read_text()
        self.asset_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        if model_path is None:
            # Playground's XML has speed-oriented solver/contact overrides.
            # Restore canonical dm_control physics, rather than silently claim parity.
            tree = ET.fromstring(xml)
            option = tree.find("option")
            option.attrib.clear()
            option.set("timestep", "0.0025")
            for child in list(option):
                option.remove(child)
            custom = tree.find("custom")
            if custom is not None:
                for child in list(custom):
                    if child.get("name") in ("max_contact_points", "max_geom_pairs"):
                        custom.remove(child)
            xml = ET.tostring(tree, encoding="unicode")
        self.xml_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.reset_mode, self.max_steps, self.steps = reset_mode, max_steps, 0
        if reset_mode not in ("random", "standing"):
            raise ValueError("reset_mode must be random (DM task) or standing (BRO bootstrap)")
        if model_path is None:
            self.model.geom_friction[:, 0] = friction
            self.model.body_mass[1:] *= mass_ratio
            self.model.body_inertia[1:] *= mass_ratio
        mujoco.mj_setConst(self.model, self.data)
        names = (
            "rootx",
            "rootz",
            "rooty",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_hip",
            "left_knee",
            "left_ankle",
        )
        joints = [self.model.joint(name) for name in names]
        self.qorder = np.array([int(j.qposadr[0]) for j in joints])
        self.vorder = np.array([int(j.dofadr[0]) for j in joints])
        self.torso = self.model.body("torso").id
        self.height_offset = float(
            self.model.body_pos[self.torso, 2] - self.model.qpos0[self.qorder[1]]
        )
        self.velocity_sensor = int(self.model.sensor("torso_subtreelinvel").adr[0])
        self._renderer = None

    def state_from_native(self, qpos, qvel):
        q = np.asarray(qpos)[..., self.qorder].copy()
        q[..., 1] += self.height_offset
        return np.concatenate((q, np.asarray(qvel)[..., self.vorder]), axis=-1)

    def native_from_state(self, state):
        state = np.asarray(state)
        qpos, qvel = np.empty_like(state[..., :9]), np.empty_like(state[..., 9:])
        q = state[..., :9].copy()
        q[..., 1] -= self.height_offset
        qpos[..., self.qorder], qvel[..., self.vorder] = q, state[..., 9:]
        return qpos, qvel

    def physical_state(self):
        return self.state_from_native(self.data.qpos, self.data.qvel)

    def set_state(self, state):
        qpos, qvel = self.native_from_state(state)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0

    def reset(self, seed):
        mujoco.mj_resetData(self.model, self.data)
        if self.reset_mode == "random":
            rng = np.random.RandomState(seed)
            for i in range(self.model.njnt):
                adr = self.model.jnt_qposadr[i]
                if self.model.jnt_limited[i]:
                    self.data.qpos[adr] = rng.uniform(*self.model.jnt_range[i])
                elif self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                    self.data.qpos[adr] = rng.uniform(-np.pi, np.pi)
        else:
            rng = np.random.default_rng(seed)
            self.data.qpos[2:] += rng.uniform(-0.005, 0.005, 7)
            # Knees cannot extend past zero; standing is an explicitly different reset.
            for i in range(self.model.njnt):
                if self.model.jnt_limited[i]:
                    adr = self.model.jnt_qposadr[i]
                    self.data.qpos[adr] = np.clip(self.data.qpos[adr], *self.model.jnt_range[i])
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.physical_state()

    def observation(self):
        return np.concatenate(
            (
                self.data.xmat[1:, [0, 2]].ravel(),
                self.data.xpos[self.torso, 2:3],
                self.data.qvel.copy(),
            )
        )

    def reward(self):
        height = self.data.xpos[self.torso, 2]
        standing = np.exp(np.log(0.1) * (max(1.2 - height, 0.0) / 0.6) ** 2)
        upright = (1 + self.data.xmat[self.torso, 8]) / 2
        vx = self.data.sensordata[self.velocity_sensor]
        move = np.clip(vx, 0.0, 1.0)
        return float((3 * standing + upright) / 4 * (5 * move + 1) / 6)

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("expected six finite actuator commands")
        applied = np.clip(action, -1, 1)
        # dm_control's Physics.step uses step1/step2 to keep position-derived
        # quantities aligned with qpos; match that endpoint convention explicitly.
        self.data.ctrl[:] = applied
        mujoco.mj_step2(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip - 1)
        mujoco.mj_step1(self.model, self.data)
        self.steps += 1
        state = self.physical_state()
        terminated = not np.isfinite(state).all()
        return state, applied, self.reward(), terminated, self.steps >= self.max_steps

    def render(self, width=640, height=480):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="side")
        return self._renderer.render()

    def metadata(self):
        physical = dict(
            friction=self.model.geom_friction.tolist(),
            mass=self.model.body_mass.tolist(),
            inertia=self.model.body_inertia.tolist(),
        )
        return dict(
            domain=self.domain,
            task="walker/walk",
            timestep=self.dt,
            integrator_timestep=float(self.model.opt.timestep),
            frame_skip=self.frame_skip,
            state="root_x, world_torso_height, root_pitch, six joint angles; "
            "root_vx, root_vz, root_pitch_rate, six joint rates",
            native_qpos_order=[self.model.joint(i).name for i in range(9)],
            height_offset=self.height_offset,
            observation_dimension=24,
            observation="body xx/xz orientations (14), torso world height, native qvel (9)",
            reset_mode=self.reset_mode,
            rl_termination="nonfinite only; falls are recoverable",
            physics=physical,
            physics_sha256=hashlib.sha256(
                json.dumps(physical, sort_keys=True).encode()
            ).hexdigest(),
            xml_sha256=self.xml_sha256,
            mujoco_version=mujoco.__version__,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()


@struct.dataclass
class State:
    data: object
    obs: jax.Array
    physical: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    steps: jax.Array
    rng: jax.Array


class MJXWalker:
    dt = 0.025
    frame_skip = 10

    def __init__(
        self,
        *,
        impl="jax",
        friction=0.7,
        mass_ratio=1.0,
        reset_mode="random",
        max_steps=1000,
        naconmax=50000,
        njmax=100,
    ):
        from mujoco import mjx

        self.native = NativeWalker(
            friction=friction, mass_ratio=mass_ratio, reset_mode=reset_mode, max_steps=max_steps
        )
        self.model = self.native.model
        self.impl, self.reset_mode, self.max_steps = impl, reset_mode, max_steps
        self.mx = mjx.put_model(self.model, impl=impl)
        kwargs = dict(impl=impl)
        if impl == "warp":
            kwargs.update(naconmax=naconmax, njmax=njmax)
        self.template = mjx.make_data(self.model, **kwargs)
        self.qorder, self.vorder = jnp.asarray(self.native.qorder), jnp.asarray(self.native.vorder)
        self.lowers, self.uppers = (
            jnp.asarray(self.model.jnt_range[:, 0]),
            jnp.asarray(self.model.jnt_range[:, 1]),
        )
        self.qzero = jnp.asarray(self.model.qpos0)

    def physical(self, data):
        q = data.qpos[self.qorder].at[1].add(self.native.height_offset)
        return jnp.concatenate((q, data.qvel[self.vorder]))

    def observations(self, data):
        orientations = data.xmat[1:].reshape(-1, 9)[:, jnp.array([0, 2])].ravel()
        return jnp.concatenate((orientations, data.xpos[self.native.torso, 2:3], data.qvel))

    def reward(self, data):
        height = data.xpos[self.native.torso, 2]
        distance = jnp.maximum(1.2 - height, 0) / 0.6
        standing = jnp.exp(jnp.log(0.1) * distance**2)
        upright = (1 + data.xmat[self.native.torso].reshape(9)[8]) / 2
        stand = (3 * standing + upright) / 4
        vx = data.sensordata[self.native.velocity_sensor]
        move = jnp.clip(1 - jnp.maximum(1 - vx, 0), 0, 1)
        return stand * (5 * move + 1) / 6

    def reset(self, key):
        from mujoco import mjx

        key, pose_key = jax.random.split(key)
        if self.reset_mode == "random":
            pose = jax.random.uniform(pose_key, (9,), minval=self.lowers, maxval=self.uppers)
            # Unbounded slides stay at qpos0, unlimited root hinge is uniform [-pi,pi].
            pose = pose.at[:2].set(self.qzero[:2])
            pose = pose.at[2].set(
                jax.random.uniform(
                    jax.random.fold_in(pose_key, 1), (), minval=-jnp.pi, maxval=jnp.pi
                )
            )
        else:
            pose = self.qzero.at[2:].add(
                jax.random.uniform(pose_key, (7,), minval=-0.005, maxval=0.005)
            )
            pose = pose.at[3:].set(jnp.clip(pose[3:], self.lowers[3:], self.uppers[3:]))
        data = mjx.forward(self.mx, self.template.replace(qpos=pose))
        return State(
            data,
            self.observations(data),
            self.physical(data),
            jnp.zeros(()),
            jnp.array(False),
            jnp.array(False),
            jnp.array(0, jnp.int32),
            key,
        )

    def step(self, state, action):
        from mujoco import mjx

        data = state.data.replace(ctrl=jnp.clip(action, -1, 1))
        data = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: mjx.step(self.mx, d), data)
        # Native dm_control forwards endpoint kinematics after integration.
        data = mjx.forward(self.mx, data)
        physical, obs = self.physical(data), self.observations(data)
        terminal = ~jnp.isfinite(physical).all()
        reward = jnp.where(terminal, 0.0, self.reward(data))
        count = state.steps + 1
        return State(
            data, obs, physical, reward, terminal, count >= self.max_steps, count, state.rng
        )

    def reset_batch(self, keys):
        return jax.vmap(self.reset)(keys)

    def step_batch(self, state, actions):
        return jax.vmap(self.step)(state, actions)

    def autoreset(self, state):
        done = state.terminated | state.truncated

        def do_reset(current):
            resets = self.reset_batch(current.rng)
            data = jax.vmap(lambda a, b, c: a.where(c, b))(current.data, resets.data, done)

            def choose(a, b):
                return jnp.where(done.reshape((*done.shape,) + (1,) * (a.ndim - done.ndim)), b, a)

            return State(
                data,
                choose(current.obs, resets.obs),
                choose(current.physical, resets.physical),
                choose(current.reward, resets.reward),
                choose(current.terminated, resets.terminated),
                choose(current.truncated, resets.truncated),
                choose(current.steps, resets.steps),
                choose(current.rng, resets.rng),
            )

        return jax.lax.cond(jnp.any(done), do_reset, lambda x: x, state)

    def close(self):
        self.native.close()
