"""Explicit immutable dynamic-ST parameters and deterministic integration."""

from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
from .physics import vehicle_dynamics_st

STATE_FIELDS = ("x", "y", "steering_angle", "speed", "yaw", "yaw_rate", "slip_angle")
FRENET_FIELDS = ("s", "ey", "steering_angle", "speed", "heading_error", "yaw_rate", "slip_angle")
CONTROL_FIELDS = ("steering_speed", "acceleration")
PARAMETER_FIELDS = (
    "mu",
    "C_Sf",
    "C_Sr",
    "lf",
    "lr",
    "h",
    "m",
    "I",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_switch",
    "a_max",
    "v_min",
    "v_max",
    "width",
    "length",
)


@struct.dataclass
class STModel:
    params: jax.Array
    action_scale: jax.Array
    timestep: float = struct.field(pytree_node=False, default=0.1)
    integrator_timestep: float = struct.field(pytree_node=False, default=0.02)

    @classmethod
    def from_vehicle(cls, vehicle, simulation):
        dtype = getattr(jnp, simulation.dtype)
        return cls(
            jnp.asarray([vehicle[k] for k in PARAMETER_FIELDS], dtype=dtype),
            jnp.asarray(simulation.action_scale, dtype=dtype),
            simulation.timestep,
            simulation.integrator_timestep,
        )


def rk4(state, control, params, dt):
    k1 = vehicle_dynamics_st(state, control, params)
    k2 = vehicle_dynamics_st(state + k1 * (dt / 2), control, params)
    k3 = vehicle_dynamics_st(state + k2 * (dt / 2), control, params)
    k4 = vehicle_dynamics_st(state + k3 * dt, control, params)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)


@jax.jit
def step(model: STModel, state, action):
    if state.shape != (7,) or action.shape != (2,):
        raise ValueError("ST step expects state (7,) and normalized action (2,)")
    state = jnp.asarray(state, dtype=model.params.dtype)
    control = jnp.asarray(action, dtype=model.params.dtype) * model.action_scale
    count = round(model.timestep / model.integrator_timestep)
    return jax.lax.fori_loop(
        0, count, lambda _, x: rk4(x, control, model.params, model.integrator_timestep), state
    )


@partial(jax.jit, static_argnames=("scan",))
def rollout(model, state, actions, scan=True):
    state = jnp.asarray(state, dtype=model.params.dtype)
    if scan:

        def advance(x, action):
            y = step(model, x, action)
            return y, y

        return jax.lax.scan(advance, state, actions)[1]
    states = []
    for action in actions:
        state = step(model, state, action)
        states.append(state)
    return jnp.stack(states)
