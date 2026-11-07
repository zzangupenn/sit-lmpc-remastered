"""Shared state/action context and timing conventions for both learners."""

import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class InferenceState:
    params: object
    apply_fn: object = struct.field(pytree_node=False)


def inference_snapshot(snapshot):
    """Drop optimizer/step state from control inputs; never change the published snapshot."""
    state = getattr(snapshot, "state", None)
    if (state is None or isinstance(state, InferenceState)
            or not hasattr(state, "params") or not hasattr(snapshot, "replace")):
        return snapshot
    return snapshot.replace(state=InferenceState(state.params, state.apply_fn))

STATE_FEATURES = ("s", "ey", "steering_angle", "speed", "heading_error")
ACTION_FEATURES = ("steering_speed", "acceleration")
ACTION_UNITS = ("rad/s", "m/s^2")


def feature_names(mode):
    if mode not in ("state", "state_action"):
        raise ValueError("input_mode must be state or state_action")
    return STATE_FEATURES + (ACTION_FEATURES if mode == "state_action" else ())


def features(states, mode="state", actions=None):
    """Actions are physical inputs applied NEXT at each supplied state."""
    if states.ndim != 2 or states.shape[1] < 5:
        raise ValueError("expected a batch of at least five Frenet state features")
    if mode == "state":
        return states[:, :5]
    feature_names(mode)
    if actions is None or actions.shape != (len(states), 2):
        raise ValueError("state_action inputs require aligned physical (N, 2) controls")
    concatenate = np.concatenate if isinstance(states, np.ndarray) else jnp.concatenate
    return concatenate((states[:, :5], actions), axis=1)


def input_metadata(mode):
    return {
        "input_mode": mode,
        "features": list(feature_names(mode)),
        "state_units": ["m", "m", "rad", "m/s", "rad"],
        "action_units": list(ACTION_UNITS) if mode == "state_action" else [],
        "action_timing": "next action at pre-action state" if mode == "state_action" else None,
    }
