"""Initial safe-set snapshots, separate from model checkpoint resume."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .dynamics import TransitionBatch, trajectory_transitions


@dataclass(frozen=True)
class InitialSnapshot:
    frenet: tuple[np.ndarray, ...]
    cartesian: tuple[np.ndarray, ...]
    state: np.ndarray
    actions: np.ndarray
    metadata: dict
    controls: tuple[np.ndarray, ...] = ()
    control_masks: tuple[np.ndarray, ...] = ()
    transitions: TransitionBatch | None = None
    density_training: dict[str, np.ndarray] | None = None


def validate_snapshot(snapshot, config):
    n = len(snapshot.frenet)
    if n < config.planner.safe_set_size or n != len(snapshot.cartesian):
        raise ValueError("initial snapshot has insufficient/mismatched safe-set laps")
    for frenet, cartesian in zip(snapshot.frenet, snapshot.cartesian, strict=True):
        if (
            frenet.ndim != 2
            or frenet.shape[1] != 8
            or frenet.shape != cartesian.shape
            or len(frenet) < 6
        ):
            raise ValueError("initial safe-set records must be matching (N, 8) arrays")
        if not np.isfinite(frenet).all() or not np.isfinite(cartesian).all():
            raise ValueError("initial safe set contains nonfinite values")
    if (
        snapshot.state.shape != (7,)
        or not np.isfinite(snapshot.state).all()
        or snapshot.state[3] <= 0
    ):
        raise ValueError("initial state must contain seven finite ST components and positive speed")
    if (
        snapshot.actions.shape != (config.planner.horizon, 2)
        or not np.isfinite(snapshot.actions).all()
        or np.any(np.abs(snapshot.actions) > 1)
    ):
        raise ValueError("initial action horizon must be finite normalized (horizon, 2) actions")
    if snapshot.controls:
        if len(snapshot.controls) != n or len(snapshot.control_masks) != n:
            raise ValueError("initial action records must match safe-set records")
        for record, control, mask in zip(
            snapshot.frenet, snapshot.controls, snapshot.control_masks, strict=True
        ):
            if (
                control.shape != (len(record), 2)
                or mask.shape != (len(record),)
                or mask.dtype != bool
            ):
                raise ValueError("initial aligned controls/masks have invalid shapes")
            if not np.isfinite(control).all():
                raise ValueError("initial aligned controls must be finite")
    if config.value.input_mode == "state_action" and (
        not snapshot.controls or not any(mask.any() for mask in snapshot.control_masks)
    ):
        raise ValueError(
            "state-action quick-load requires aligned controls; disable quick_load to collect fresh warm-up laps"
        )
    if snapshot.transitions is not None:
        batch, n = snapshot.transitions, len(snapshot.transitions)
        shapes = {"states": (n, 7), "next_states": (n, 7), "controls": (n, 2),
                  "intervals": (n,), "substeps": (n,), "ids": (n, 2)}
        if any(getattr(batch, key).shape != shape or not np.isfinite(getattr(batch, key)).all()
               for key, shape in shapes.items()) or np.any(batch.intervals <= 0) or np.any(batch.substeps <= 0):
            raise ValueError("initial physical transition records must be finite and aligned")
    if snapshot.density_training is not None and snapshot.transitions is None:
        raise ValueError("initial NF training history requires original measured transitions")
    return snapshot


def save_initial(path, safe_set, state, actions, config, *, trajectories=(), density_training=None):
    transitions = TransitionBatch.concatenate(
        trajectory_transitions(trajectory, config.simulation.integrator_timestep)
        for trajectory in trajectories
    )
    snapshot = InitialSnapshot(
        tuple(safe_set.frenet),
        tuple(safe_set.cartesian),
        np.asarray(state),
        np.asarray(actions),
        {"schema": 4 if density_training is not None else 3 if len(transitions) else 2 if safe_set.controls else 1,
         "config": config.as_dict()},
        tuple(safe_set.controls),
        tuple(safe_set.control_masks),
        transitions if len(transitions) else None,
        density_training,
    )
    validate_snapshot(snapshot, config)
    payload = {
        "metadata": np.asarray(json.dumps(snapshot.metadata)),
        "state": snapshot.state,
        "actions": snapshot.actions,
    }
    if snapshot.transitions is not None:
        payload.update({f"dynamics/{name}": getattr(snapshot.transitions, name)
                        for name in TransitionBatch.__dataclass_fields__})
    if density_training is not None:
        payload.update({f"density_training/{name}": value for name, value in density_training.items()})
    for i, (frenet, cartesian) in enumerate(zip(snapshot.frenet, snapshot.cartesian, strict=True)):
        payload[f"safe_set_{i:04d}/frenet"] = frenet
        payload[f"safe_set_{i:04d}/cartesian"] = cartesian
        if snapshot.controls:
            payload[f"safe_set_{i:04d}/controls"] = snapshot.controls[i]
            payload[f"safe_set_{i:04d}/control_mask"] = snapshot.control_masks[i]
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite initial snapshot: {path}")
    np.savez_compressed(path, **payload)


def load_initial(path, config, trust_legacy=False):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        native = "metadata" in data.files
        if native:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("schema") not in (1, 2, 3, 4):
                raise ValueError("unsupported initial snapshot schema")
            old = metadata["config"]
            new = config.as_dict()
            for section, fields in {
                "simulation": (
                    "model",
                    "map",
                    "friction",
                    "mass_ratio",
                    "timestep",
                    "integrator_timestep",
                    "dtype",
                    "action_scale",
                ),
                "planner": ("friction", "horizon", "half_width", "extension_seconds"),
            }.items():
                for field in fields:
                    # JSON canonicalizes tuple/list fields from both sides.
                    previous = (
                        old[section].get(field, 1.0)
                        if section == "simulation" and field == "mass_ratio"
                        else old[section][field]
                    )
                    if json.dumps(previous) != json.dumps(new[section][field]):
                        raise ValueError(
                            f"Initial snapshot configuration differs: {section}.{field}"
                        )
            prefixes = sorted(
                {key.split("/")[0] for key in data.files if key.startswith("safe_set_")}
            )
            snapshot = InitialSnapshot(
                tuple(data[f"{p}/frenet"].copy() for p in prefixes),
                tuple(data[f"{p}/cartesian"].copy() for p in prefixes),
                data["state"].copy(),
                data["actions"].copy(),
                metadata,
                tuple(data[f"{p}/controls"].copy() for p in prefixes)
                if metadata["schema"] >= 2
                else (),
                tuple(data[f"{p}/control_mask"].copy() for p in prefixes)
                if metadata["schema"] >= 2
                else (),
                TransitionBatch(**{name: data[f"dynamics/{name}"].copy()
                                   for name in TransitionBatch.__dataclass_fields__})
                if metadata["schema"] >= 3 else None,
                {name: data[f"density_training/{name}"].copy()
                 for name in ("states", "ids", "episode_ids")}
                if metadata["schema"] >= 4 else None,
            )
    if not native:
        if not trust_legacy:
            raise ValueError(
                "Legacy init files may execute pickle payloads; only enable trust_legacy_init for your own trusted file"
            )
        # Explicit opt-in only. Never attempt to unpickle during format detection.
        with np.load(path, allow_pickle=True) as data:
            required = {"safe_set", "safe_set_xy", "init_state", "a_opt_records"}
            if not required <= set(data.files):
                raise ValueError("legacy init file is missing required SIT/TWIT records")

            def record(name):
                value = data[name]
                # utilitySuite stores each record as a scalar {name: value} object.
                if value.shape == () and value.dtype == object:
                    value = value.item()
                    if isinstance(value, dict):
                        value = value[name]
                return value

            actions = np.asarray(record("a_opt_records")[-1][-1], dtype=config.simulation.dtype)[
                : config.planner.horizon
            ]
            snapshot = InitialSnapshot(
                tuple(np.asarray(a, dtype=config.simulation.dtype) for a in record("safe_set")),
                tuple(np.asarray(a, dtype=config.simulation.dtype) for a in record("safe_set_xy")),
                np.asarray(record("init_state"), dtype=config.simulation.dtype).reshape(-1),
                actions,
                {
                    "schema": "trusted-legacy",
                    "source": str(path),
                    "compatibility": "shape-only; legacy files lack full physical metadata",
                },
            )
    return validate_snapshot(snapshot, config)


def restore_safe_set(safe_set, snapshot):
    count = len(snapshot.frenet) if safe_set.config.safe_set_min_states else safe_set.config.safe_set_size
    safe_set.frenet = [a.copy() for a in snapshot.frenet[-count:]]
    safe_set.cartesian = [a.copy() for a in snapshot.cartesian[-count:]]
    safe_set.controls = [a.copy() for a in snapshot.controls[-count:]]
    safe_set.control_masks = [a.copy() for a in snapshot.control_masks[-count:]]
    safe_set.state_ids = []  # Imported arrays do not establish original state identities.
    if safe_set.config.safe_set_query != "nf_density":
        safe_set.trim()
    for array in (
        *safe_set.frenet,
        *safe_set.cartesian,
        *safe_set.controls,
        *safe_set.control_masks,
    ):
        array.setflags(write=False)
    # SIT quick loading starts with no pending raw lap for the extension pair.
    safe_set.previous = None
    safe_set.previous_trajectory = None
