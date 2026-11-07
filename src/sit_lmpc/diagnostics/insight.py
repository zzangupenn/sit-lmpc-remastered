"""Bounded, numeric-only planner recordings; replay never reruns a policy."""

from collections import deque
from pathlib import Path
import json
import os

import numpy as np


def _encode(value, arrays, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise ValueError("insight arrays must be numeric, without Python objects")
        if id(value) in memo:
            return {"array": memo[id(value)]}
        key = f"array_{len(arrays):06d}"
        memo[id(value)] = key
        arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays, memo)
    if isinstance(value, float) and not np.isfinite(value):
        # Retain scalar temporaries until serialization is complete: their ids
        # must not be recycled into the array-reference memo.
        scalar = np.asarray(value)
        return _encode(scalar, arrays, memo)
    if isinstance(value, dict):
        return {key: _encode(item, arrays, memo) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, arrays, memo) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported insight value: {type(value).__name__}")


def _decode(value, archive, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, dict):
        if set(value) == {"array"}:
            key = value["array"]
            if key not in memo:
                memo[key] = archive[key].copy()
            return memo[key]
        return {key: _decode(item, archive, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, archive, memo) for item in value]
    return value


def load_insight(path):
    """Load our versioned NPZ using allow_pickle=False, including invalid samples."""
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as archive:
        if "manifest" not in archive:
            raise ValueError("Expected a BRO insight.npz recording (SIT object archives are a different format)")
        manifest = json.loads(str(archive["manifest"]))
        if manifest.get("schema") not in (1, 2) or manifest.get("kind") != "sit_lmpc_insight":
            raise ValueError("unsupported BRO insight recording schema")
        result = _decode(manifest["recording"], archive)
    if not result["steps"]:
        raise ValueError("insight recording contains no planner decisions")
    return result


class InsightRecorder:
    def __init__(self, directory, steps, config, emit=lambda *args, **kwargs: None):
        self.directory, self.config, self.emit = Path(directory), config, emit
        self.steps = deque(maxlen=steps)
        self.history = deque(maxlen=steps)
        self.contexts = {}

    @property
    def enabled(self):
        return bool(self.steps.maxlen)

    def record(self, result, *, step, attempt, lap, time, mode, cartesian, frenet, actions, context=None):
        if not self.enabled:
            return
        # Debug arrays have already crossed the device boundary. Own every leaf:
        # later planner updates or GUI work must not change a recorded decision.
        import jax

        record = dict(
            step=int(step), attempt=int(attempt), lap=int(lap), time=float(time), mode=mode,
            state=np.asarray(cartesian), frenet=np.asarray(frenet), action=np.asarray(result.action),
            actions=np.asarray(actions), trajectory=np.asarray(result.trajectory),
            reference=None if result.reference is None else np.asarray(result.reference),
            lambdas=result.lambdas, safe_set_violation=result.safe_set_violation,
            boundary_violation=result.boundary_violation, metrics=result.metrics,
            debug=result.debug, risk=result.risk, applied=False,
        )
        self.steps.append(jax.tree.map(lambda x: x.copy() if isinstance(x, np.ndarray) else x, record))
        self.steps[-1]["views"] = {}
        self.contexts[int(step)] = context
        retained = {item["step"] for item in (*self.history, *self.steps)}
        self.contexts = {key: value for key, value in self.contexts.items() if key in retained}

    def applied(self, cartesian, frenet, control):
        if self.enabled and self.steps:
            if not self.steps[-1]["applied"]:
                self.history.append(self.steps[-1])
            self.steps[-1].update(applied=True, next_state=np.asarray(cartesian).copy(),
                                  next_frenet=np.asarray(frenet).copy(), control=np.asarray(control).copy())

    def prediction(self, state):
        if self.enabled and self.steps:
            self.steps[-1]["predicted"] = np.asarray(state).copy()

    def display(self, observation, overlays=None):
        if self.enabled and self.steps:
            from copy import deepcopy

            self.steps[-1]["render_observation"] = deepcopy(observation)
            if overlays is not None:
                self.steps[-1]["overlays"] = {k: np.asarray(v).copy() for k, v in overlays.items()}

    def view(self, frame, number):
        views = frame.setdefault("views", {})
        key = str(number)
        if key not in views:
            context = self.contexts.get(frame["step"])
            if context is None:
                views[key] = dict(unavailable="This recording has no historical context for this view.")
            else:
                try:
                    views[key] = context.get(frame, number)
                except Exception as error:
                    views[key] = dict(unavailable=f"Historical view unavailable: {error}")
        return views[key]

    def save(self, reason, *, crash=False, materialize=True, pump=lambda: None):
        if not self.steps:
            return None
        arrays = {}
        if materialize:
            for frame in self.steps:
                for number in range(2, 7):
                    self.view(frame, number)
                pump()
        recording = dict(reason=reason, config=self.config.as_dict(), steps=list(self.steps))
        manifest = dict(schema=2, kind="sit_lmpc_insight", recording=_encode(recording, arrays))
        arrays["manifest"] = np.asarray(json.dumps(manifest, allow_nan=False))
        paths = [self.directory / "insight.npz"]
        if crash:
            paths.append(self.directory / "insight_before_crash.npz")
        for path in paths:
            pending = path.with_name(f".{path.name}.pending")
            try:
                with pending.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(pending, path)
            finally:
                pending.unlink(missing_ok=True)
        self.emit("insight_saved", file=paths[-1].name, steps=len(self.steps), reason=reason)
        return paths[-1]
