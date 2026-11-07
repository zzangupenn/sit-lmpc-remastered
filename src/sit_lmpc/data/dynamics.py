"""Aligned physical transitions and experiment-local recent/history selection."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    controls: np.ndarray
    next_states: np.ndarray
    intervals: np.ndarray
    substeps: np.ndarray
    ids: np.ndarray

    def __len__(self):
        return len(self.controls)

    def take(self, indices):
        return TransitionBatch(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )

    @classmethod
    def concatenate(cls, batches):
        batches = list(batches)
        if not batches:
            return cls(
                np.empty((0, 7)),
                np.empty((0, 2)),
                np.empty((0, 7)),
                np.empty(0),
                np.empty(0),
                np.empty((0, 2), dtype=np.int64),
            )
        return cls(
            **{
                name: np.concatenate([getattr(batch, name) for batch in batches])
                for name in cls.__dataclass_fields__
            }
        )


def trajectory_transitions(trajectory, fallback_substep):
    """No pair crosses an attempt/reset boundary; contacts never train ST."""
    intervals = np.diff(trajectory.times)
    substep = trajectory.integrator_timestep or fallback_substep
    ratios = intervals / substep
    valid = (
        np.isfinite(trajectory.cartesian[:-1]).all(axis=1)
        & np.isfinite(trajectory.cartesian[1:]).all(axis=1)
        & np.isfinite(trajectory.controls).all(axis=1)
        & np.isfinite(ratios)
        & (ratios >= 1 - 1e-6)
        & np.isclose(ratios, np.rint(ratios), rtol=1e-5, atol=1e-5)
    )
    if trajectory.contacts is not None:
        valid &= ~trajectory.contacts
    if trajectory.termination_reason == "collision" and len(valid):
        valid[-1] = False
    indices = np.flatnonzero(valid)
    return TransitionBatch(
        trajectory.cartesian[:-1][indices].copy(),
        trajectory.controls[indices].copy(),
        trajectory.cartesian[1:][indices].copy(),
        intervals[indices].copy(),
        np.full(len(indices), substep),
        np.column_stack((np.full(len(indices), trajectory.attempt_id), trajectory.steps[indices])),
    )


def initial_transitions(initial, fallback_timestep, fallback_substep):
    """Prefer original physical records; conservatively read old safe-set snapshots.

    Safe-set extension rows are not necessarily consecutive observations. The
    reward/time column preserves within-section time differences in old files;
    never interpret a skipped section or a reset as a measured transition.
    """
    if getattr(initial, "transitions", None) is not None:
        return initial.transitions
    if not initial.controls:
        return TransitionBatch.concatenate(())
    sim = initial.metadata.get("config", {}).get("simulation", {})
    dt = sim.get("timestep", fallback_timestep)
    substep = sim.get("integrator_timestep", fallback_substep)
    batches = []
    for episode, (states, controls, mask) in enumerate(zip(
        initial.cartesian, initial.controls, initial.control_masks, strict=True
    )):
        intervals = np.diff(states[:, -1])
        displacement = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
        bound = np.maximum(np.abs(states[:-1, 3]), np.abs(states[1:, 3])) * dt
        valid = (mask[:-1] & np.isfinite(states[:-1, :7]).all(axis=1)
                 & np.isfinite(states[1:, :7]).all(axis=1)
                 & np.isfinite(controls[:-1]).all(axis=1)
                 & np.isclose(intervals, dt, rtol=1e-4, atol=1e-5)
                 & (displacement <= bound + 1e-3))
        indices = np.flatnonzero(valid)
        batches.append(TransitionBatch(states[:-1, :7][indices], controls[:-1][indices],
            states[1:, :7][indices], np.full(len(indices), dt), np.full(len(indices), substep),
            np.column_stack((np.full(len(indices), -episode - 1), indices))))
    result = TransitionBatch.concatenate(batches)
    if len(result):
        # Only the imported extension duplicates are deduplicated, not ordinary
        # repeated measurements from different trials under the same dynamics.
        _, unique = np.unique(np.column_stack((result.states, result.controls, result.next_states)),
                              axis=0, return_index=True)
        result = result.take(np.sort(unique))
    return result


@dataclass(frozen=True)
class RolloutBatch:
    """Recorded control windows; masks stop at gaps, contacts, and attempt ends."""

    states: np.ndarray
    controls: np.ndarray
    targets: np.ndarray
    intervals: np.ndarray
    substeps: np.ndarray
    mask: np.ndarray

    def __len__(self):
        return len(self.states)


@dataclass(frozen=True)
class ReplaySelection:
    transitions: TransitionBatch
    weights: np.ndarray
    recent_count: int
    history_count: int
    windows: RolloutBatch | None = None


class DynamicsReplay:
    """Data selection is independent of the inference backend and outcome labels."""

    def __init__(self):
        self.episodes = {}
        self.last_selection = np.empty((0, 2), dtype=np.int64)

    def add(self, attempt_id, outcome, batch):
        if attempt_id in self.episodes:
            raise ValueError(f"dynamics replay already contains attempt {attempt_id}")
        self.episodes[attempt_id] = (outcome, batch)

    def windows(self, selection, horizon):
        """Use selected starts, but fetch continuations from their full episodes.

        Sparse historical sampling must not join unrelated transitions. Each
        start retains its replay weight regardless of how much horizon exists.
        """
        count = len(selection.transitions)
        states = selection.transitions.states.copy()
        controls = np.zeros((count, horizon, 2))
        targets = np.zeros((count, horizon, 7))
        intervals = np.full((count, horizon), 0.1)
        substeps = np.full((count, horizon), 0.02)
        mask = np.zeros((count, horizon), dtype=bool)
        lookups = {}
        for row, (attempt_id, step_id) in enumerate(selection.transitions.ids):
            attempt_id = int(attempt_id)
            batch = self.episodes[attempt_id][1]
            if attempt_id not in lookups:
                lookups[attempt_id] = {int(identity[1]): i for i, identity in enumerate(batch.ids)}
            start = lookups[attempt_id][int(step_id)]
            for offset in range(min(horizon, len(batch) - start)):
                index = start + offset
                if offset and (
                    batch.ids[index, 1] != batch.ids[index - 1, 1] + 1
                    or not np.array_equal(batch.states[index], batch.next_states[index - 1])
                ):
                    break
                controls[row, offset] = batch.controls[index]
                targets[row, offset] = batch.next_states[index]
                intervals[row, offset] = batch.intervals[index]
                substeps[row, offset] = batch.substeps[index]
                mask[row, offset] = True
        return RolloutBatch(states, controls, targets, intervals, substeps, mask)

    def select(self, rng, history_samples, recent_fraction):
        recent_ids = {
            max(
                (
                    key
                    for key, (outcome, batch) in self.episodes.items()
                    if outcome == wanted and len(batch)
                ),
                default=None,
            )
            for wanted in ("success", "failure")
        } - {None}
        recent = TransitionBatch.concatenate(self.episodes[key][1] for key in sorted(recent_ids))
        old = [batch for key, (_, batch) in sorted(self.episodes.items()) if key not in recent_ids]
        total = sum(map(len, old))
        if total > history_samples:
            # Sample archive offsets first: do not copy the full growing archive
            # just to retain a small historical batch.
            selected = np.sort(rng.choice(total, history_samples, replace=False))
            chunks, offset = [], 0
            for episode in old:
                positions = (
                    selected[(selected >= offset) & (selected < offset + len(episode))] - offset
                )
                if len(positions):
                    chunks.append(episode.take(positions))
                offset += len(episode)
            history = TransitionBatch.concatenate(chunks)
        else:
            history = TransitionBatch.concatenate(old)
        nr, nh = len(recent), len(history)
        fraction = recent_fraction if nr and nh else 1.0 if nr else 0.0
        weights = np.concatenate(
            (
                np.full(nr, fraction / nr) if nr else np.empty(0),
                np.full(nh, (1 - fraction) / nh) if nh else np.empty(0),
            )
        )
        # If a configured zero-weight cohort is the only source, it is still usable.
        batch = TransitionBatch.concatenate((recent, history))
        self.last_selection = batch.ids.copy()
        return ReplaySelection(batch, weights, nr, nh)
