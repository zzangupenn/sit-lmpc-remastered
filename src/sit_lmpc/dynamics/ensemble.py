"""Probability-weighted predictions of independent, fixed-model trajectories."""

from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from .bank import STModelBank, bank_rollout, bank_step
from .st import rollout, step

ANGLE_FIELDS = (2, 4, 6)


@struct.dataclass
class STEnsemble:
    bank: STModelBank
    weights: jax.Array
    weight_kind: str = struct.field(pytree_node=False, default="model_probabilities")

    @property
    def params(self):
        # Full member parameters, never an averaged friction/parameter vector.
        return self.bank.params

    @property
    def action_scale(self):
        return self.bank.action_scale

    @property
    def timestep(self):
        return self.bank.timestep

    @property
    def integrator_timestep(self):
        return self.bank.integrator_timestep


@partial(jax.jit, static_argnames=("differentiable",))
def weighted_states(states, probabilities, *, differentiable=False):
    """Reduce model axis 0; circular means retain the branch angle convention."""
    if states.shape[0] != len(probabilities) or states.shape[-1] != 7:
        raise ValueError("ensemble states must have model axis 0 and seven components")
    if len(probabilities) == 1:
        return states[0]
    probabilities = jnp.asarray(probabilities, dtype=states.dtype)
    shape = (len(probabilities),) + (1,) * (states.ndim - 1)
    coefficients = probabilities.reshape(shape)
    # A zero-probability branch must not poison the mean with NaN/Inf.
    supported = jnp.where(
        jnp.isfinite(states) if differentiable else coefficients > 0, states, 0
    )
    mean = jnp.sum(coefficients * supported, axis=0)
    reference = states[jnp.argmax(probabilities)]
    for field in ANGLE_FIELDS:
        angular_weights = probabilities.reshape(shape[:-1])
        sine = jnp.sum(angular_weights * jnp.sin(supported[..., field]), axis=0)
        cosine = jnp.sum(angular_weights * jnp.cos(supported[..., field]), axis=0)
        angle = jnp.arctan2(sine, cosine)
        anchor = reference[..., field]
        # Only the angle's 2*pi representation comes from the anchor, not its
        # physical prediction; an undefined (antipodal) mean uses it explicitly.
        lifted = anchor + jnp.arctan2(jnp.sin(angle - anchor), jnp.cos(angle - anchor))
        ambiguous = jnp.hypot(sine, cosine) <= 8 * jnp.finfo(states.dtype).eps
        mean = mean.at[..., field].set(jnp.where(ambiguous, anchor, lifted))
    # Fitting needs the derivative towards a currently zero-weight finite
    # branch, even at a simplex vertex. Planning keeps exact one-hot results.
    return mean if differentiable else jnp.where(jnp.count_nonzero(probabilities) == 1, reference, mean)


@jax.jit
def predictor_step(model, state, action):
    if hasattr(model, "step_members"):
        return weighted_states(model.step_members(state, action), model.weights)
    if isinstance(model, STEnsemble):
        return weighted_states(bank_step(model.bank, state, action), model.weights)
    return step(model, state, action)


@partial(jax.jit, static_argnames=("scan",))
def member_rollouts(model, state, actions, scan=True):
    """Keep each member's friction and its own evolving state for the full horizon."""
    if hasattr(model, "rollout_members"):
        return model.rollout_members(state, actions)
    if isinstance(model, STEnsemble):
        return bank_rollout(model.bank, state, actions, scan)
    return rollout(model, state, actions, scan)[None]


@partial(jax.jit, static_argnames=("scan",))
def rollout_prediction(model, state, actions, scan=True):
    members = member_rollouts(model, state, actions, scan)
    prediction = (
        weighted_states(members, model.weights) if hasattr(model, "weights") else members[0]
    )
    return prediction, members


@partial(jax.jit, static_argnames=("scan",))
def predictor_rollout(model, state, actions, scan=True):
    return rollout_prediction(model, state, actions, scan)[0]


def predictor_friction(model):
    """An ensemble has no single friction coefficient; never invent one in logs."""
    return None if hasattr(model, "weights") else float(model.params[0])


def predictor_context(model):
    if hasattr(model, "context"):
        return model.context()
    if isinstance(model, STEnsemble):
        return {
            "kind": "ensemble",
            "frictions": np.asarray(model.bank.params[:, 0]).tolist(),
            "weights": np.asarray(model.weights).tolist(),
            "weight_kind": model.weight_kind,
        }
    return {"kind": "single", "friction": predictor_friction(model)}
