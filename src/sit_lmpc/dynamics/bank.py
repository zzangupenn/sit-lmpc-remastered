"""Fixed-friction ST family; no belief weights or simulator ground truth."""

from functools import partial
import math
from flax import struct
import jax
import jax.numpy as jnp

from .st import STModel, PARAMETER_FIELDS, step, rollout


@struct.dataclass
class STModelBank:
    params: jax.Array
    action_scale: jax.Array
    timestep: float = struct.field(pytree_node=False)
    integrator_timestep: float = struct.field(pytree_node=False)

    @classmethod
    def from_model(cls, model, frictions):
        if not frictions or any(
            isinstance(mu, bool)
            or not isinstance(mu, (int, float))
            or not math.isfinite(mu)
            or mu <= 0
            for mu in frictions
        ):
            raise ValueError("frictions must be positive finite numbers")
        if len(set(frictions)) != len(frictions):
            raise ValueError("frictions must be unique")
        params = jnp.broadcast_to(model.params, (len(frictions), len(model.params)))
        params = params.at[:, PARAMETER_FIELDS.index("mu")].set(
            jnp.asarray(frictions, dtype=model.params.dtype)
        )
        return cls(params, model.action_scale, model.timestep, model.integrator_timestep)

    def member(self, params):
        return STModel(params, self.action_scale, self.timestep, self.integrator_timestep)


@jax.jit
def bank_step(bank, state, action):
    return jax.vmap(lambda params: step(bank.member(params), state, action))(bank.params)


@partial(jax.jit, static_argnames=("scan",))
def bank_rollout(bank, state, actions, scan=True):
    return jax.vmap(lambda params: rollout(bank.member(params), state, actions, scan))(bank.params)
