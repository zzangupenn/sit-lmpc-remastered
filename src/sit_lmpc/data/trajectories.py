"""Experiment-local lap attempts, with aligned transitions and explicit outcomes."""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
import numpy as np

from sit_lmpc.data.laps import Lap
from sit_lmpc.dynamics.st import STATE_FIELDS, FRENET_FIELDS, CONTROL_FIELDS

OUTCOMES = ("success", "failure", "incomplete")
ARRAY_FIELDS = (
    "cartesian",
    "frenet",
    "controls",
    "times",
    "progress",
    "physical_failure",
    "goal",
    "controllers",
    "value_generations",
    "steps",
)


@dataclass(frozen=True)
class Trajectory:
    attempt_id: int
    lap: int
    outcome: str
    termination_reason: str
    cartesian: np.ndarray
    frenet: np.ndarray
    controls: np.ndarray
    times: np.ndarray
    progress: np.ndarray
    physical_failure: np.ndarray
    goal: np.ndarray
    controllers: np.ndarray
    value_generations: np.ndarray
    steps: np.ndarray
    belief_generations: np.ndarray | None = None
    contacts: np.ndarray | None = None
    integrator_timestep: float | None = None
    action_scale: tuple[float, float] | None = None

    def validate(self):
        count = len(self.controls)
        for name in ("belief_generations", "contacts"):
            value = getattr(self, name)
            if value is not None and value.shape != (count,):
                raise ValueError(f"trajectory {name} must align with controls")
        if self.integrator_timestep is not None and (
            not np.isfinite(self.integrator_timestep) or self.integrator_timestep <= 0
        ):
            raise ValueError("trajectory integration interval must be positive and finite")
        if self.outcome not in OUTCOMES or self.attempt_id < 1 or self.lap < 1:
            raise ValueError("invalid trajectory identity/outcome")
        if (
            self.cartesian.shape != (count + 1, 7)
            or self.frenet.shape != self.cartesian.shape
            or self.controls.shape != (count, 2)
            or any(
                getattr(self, name).shape != (count + 1,)
                for name in ("times", "progress", "physical_failure", "goal")
            )
            or any(
                getattr(self, name).shape != (count,)
                for name in ("controllers", "value_generations", "steps")
            )
        ):
            raise ValueError("trajectory requires T+1 aligned states/times and T controls/context")
        if (
            self.times[0] != 0
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("trajectory times must start at zero and strictly increase")
        if self.outcome != "incomplete" and not all(
            np.isfinite(getattr(self, name)).all()
            for name in ("cartesian", "frenet", "controls", "progress")
        ):
            raise ValueError("completed physical outcomes require finite trajectory data")
        if np.any(self.physical_failure[:-1]) or np.any(self.goal[:-1]):
            raise ValueError("trajectory must stop at its first physical failure/success")
        if bool(self.physical_failure[-1]) != (self.outcome == "failure") or bool(
            self.goal[-1]
        ) != (self.outcome == "success"):
            raise ValueError("trajectory outcome and terminal flags disagree")
        return self

    def as_lap(self):
        """Measured post-step records for summaries and numeric lap exports."""
        return Lap(self.cartesian[1:], self.frenet[1:], self.controls, self.times[1:])

    def as_learning_lap(self, track_length):
        """Keep the crossing endpoint, lifted beyond L for safe-set/value learning."""
        if self.outcome != "success":
            raise ValueError("safe-set/value learning requires a successful trajectory")
        frenet = self.frenet[1:].copy()
        # Use the collector's accumulated progress, including the initial state,
        # rather than mistaking the wrapped terminal s for a new lap's prefix.
        frenet[:, 0] = self.frenet[0, 0] % track_length + self.progress[1:]
        return Lap(self.cartesian[1:], frenet, self.controls, self.times[1:])


class AttemptCollector:
    def __init__(self, attempt_id, lap, cartesian, frenet, track_length, *, integrator_timestep=None, action_scale=None):
        self.attempt_id, self.lap = attempt_id, lap
        self.track_length = float(track_length)
        # Keep the existing map finish-line segmentation. A sampled lap start
        # can be just beyond the line; requiring another relative L would drift
        # boundaries every lap and break the baseline safe-set extension layout.
        self.goal_progress = self.track_length - float(frenet[0]) % self.track_length
        self.cartesian, self.frenet = [np.asarray(cartesian).copy()], [np.asarray(frenet).copy()]
        self.controls, self.controllers, self.value_generations, self.steps = [], [], [], []
        self.times, self.progress = [0.0], [0.0]
        self.physical_failure, self.goal = [False], [False]
        self.belief_generations, self.contacts = [], []
        self.integrator_timestep, self.action_scale = integrator_timestep, action_scale

    def append(self, cartesian, frenet, control, time, controller, generation, step, *, belief_generation=0, contact=False):
        previous_s, s = self.frenet[-1][0], frenet[0]
        delta = (
            float(s) - float(previous_s) + self.track_length / 2
        ) % self.track_length - self.track_length / 2
        self.progress.append(self.progress[-1] + delta)
        self.cartesian.append(np.asarray(cartesian).copy())
        self.frenet.append(np.asarray(frenet).copy())
        self.controls.append(np.asarray(control).copy())
        self.times.append(float(time))
        self.physical_failure.append(False)
        self.goal.append(False)
        self.controllers.append(controller)
        self.value_generations.append(generation)
        self.steps.append(step)
        self.belief_generations.append(belief_generation)
        self.contacts.append(contact)

    @property
    def completed_lap(self):
        return self.progress[-1] >= self.goal_progress

    def finish(self, outcome, reason):
        if outcome == "success" and not self.completed_lap:
            raise ValueError("success requires forward progress to the next finish line")
        self.physical_failure[-1] = outcome == "failure"
        self.goal[-1] = outcome == "success"
        return Trajectory(
            self.attempt_id,
            self.lap,
            outcome,
            reason,
            np.asarray(self.cartesian),
            np.asarray(self.frenet),
            np.asarray(self.controls).reshape(-1, 2),
            np.asarray(self.times),
            np.asarray(self.progress),
            np.asarray(self.physical_failure),
            np.asarray(self.goal),
            np.asarray(self.controllers, dtype=str),
            np.asarray(self.value_generations, dtype=np.int64),
            np.asarray(self.steps, dtype=np.int64),
            np.asarray(self.belief_generations, dtype=np.int64),
            np.asarray(self.contacts, dtype=bool),
            self.integrator_timestep,
            self.action_scale,
        ).validate()


def save_trajectory(path, trajectory):
    trajectory.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite trajectory: {path}")
    metadata = {
        name: getattr(trajectory, name)
        for name in ("attempt_id", "lap", "outcome", "termination_reason", "integrator_timestep", "action_scale")
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".attempt-", suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                schema=np.asarray(2),
                metadata=np.asarray(json.dumps(metadata)),
                **{name: getattr(trajectory, name) for name in ARRAY_FIELDS},
                belief_generations=np.zeros(len(trajectory.controls), dtype=np.int64)
                if trajectory.belief_generations is None else trajectory.belief_generations,
                contacts=np.zeros(len(trajectory.controls), dtype=bool)
                if trajectory.contacts is None else trajectory.contacts,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as data:
        if "schema" not in data or int(data["schema"]) not in (1, 2) or "metadata" not in data:
            raise ValueError("unsupported trajectory format")
        return Trajectory(
            **json.loads(str(data["metadata"])),
            **{name: data[name].copy() for name in ARRAY_FIELDS},
            **{name: data[name].copy() for name in ("belief_generations", "contacts") if name in data},
        ).validate()


class TrajectoryDataset:
    """A single experiment's append-only success/failure/incomplete collections."""

    def __init__(self, directory, config, track_length, resume=False):
        self.directory = Path(directory) / "trajectories"
        if resume:
            if not self.directory.is_dir():
                raise FileNotFoundError(f"Missing experiment trajectories: {self.directory}")
        else:
            self.directory.mkdir()
        self.counts = dict.fromkeys(OUTCOMES, 0)
        self.applied_steps = 0
        for outcome in OUTCOMES:
            if resume:
                if not (self.directory / outcome).is_dir():
                    raise FileNotFoundError(f"Missing {outcome} trajectory collection")
            else:
                (self.directory / outcome).mkdir()
        if resume:
            for outcome in OUTCOMES:
                for path in sorted((self.directory / outcome).glob("attempt-*.npz")):
                    trajectory = load_trajectory(path)
                    if trajectory.outcome != outcome:
                        raise ValueError(f"Trajectory collection mismatch: {path}")
                    self.counts[outcome] += 1
                    self.applied_steps += len(trajectory.controls)
            return
        context = {
            "schema": 1,
            "config": config.as_dict(),
            "experiment_metadata": "../metadata.json",
            "state_fields": STATE_FIELDS,
            "frenet_fields": FRENET_FIELDS,
            "control_fields": CONTROL_FIELDS,
            "control_units": ["rad/s", "m/s^2"],
            "task": {
                "goal": "forward lap completion at the map finish line",
                "track_length": float(track_length),
                "boundary_half_width": config.planner.half_width,
            },
            "warmup_source": str(Path(config.output.init_path).expanduser().resolve())
            if config.output.quick_load
            else None,
        }
        (self.directory / "context.json").write_text(json.dumps(context, indent=2) + "\n")

    def add(self, trajectory):
        relative = Path(trajectory.outcome) / f"attempt-{trajectory.attempt_id:06d}.npz"
        path = self.directory / relative
        save_trajectory(path, trajectory)
        self.counts[trajectory.outcome] += 1
        self.applied_steps += len(trajectory.controls)
        record = {
            "attempt_id": trajectory.attempt_id,
            "lap": trajectory.lap,
            "outcome": trajectory.outcome,
            "termination_reason": trajectory.termination_reason,
            "steps": len(trajectory.controls),
            "duration": float(trajectory.times[-1]),
            "file": str(relative),
        }
        with (self.directory / "index.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return path

    @property
    def trials(self):
        """Successful and physically failed lap trials counted toward max_laps."""
        return self.counts["success"] + self.counts["failure"]


def load_trajectories(directory, outcome):
    """Read one outcome collection from one experiment; never aggregate runs."""
    if outcome not in OUTCOMES:
        raise ValueError("unknown trajectory outcome")
    return [
        load_trajectory(path)
        for path in sorted((Path(directory) / "trajectories" / outcome).glob("attempt-*.npz"))
    ]
