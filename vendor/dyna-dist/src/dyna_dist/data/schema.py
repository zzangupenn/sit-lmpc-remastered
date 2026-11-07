"""Validated arrays with explicit transition identity and fixed physical interval."""

from dataclasses import dataclass
import numpy as np
from dyna_dist.domains import get_domain


@dataclass(frozen=True)
class Transitions:
    states: np.ndarray
    controls: np.ndarray
    next_states: np.ndarray
    ids: tuple[str, ...]
    timestep: float
    domain: str = "st"
    stages: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.states)
        spec = get_domain(self.domain)
        if (
            self.states.shape != (n, spec.state_dimension)
            or self.next_states.shape != (n, spec.state_dimension)
            or self.controls.shape != (n, spec.control_dimension)
            or len(self.ids) != n
        ):
            raise ValueError(f"Transitions need {spec.name} states/targets and controls with N IDs")
        if len(set(self.ids)) != n:
            raise ValueError("transition IDs must be unique within a batch")
        if not all(np.isfinite(x).all() for x in (self.states, self.controls, self.next_states)):
            raise ValueError("transition arrays must be finite")
        if not np.isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("transition timestep must be positive")
        if self.stages is not None and (
            self.stages.shape != (n,) or self.stages.dtype.kind not in "iu"
        ):
            raise ValueError("learning stages must be an integer vector aligned with transitions")

    def __len__(self):
        return len(self.states)

    def take(self, indices):
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
        return Transitions(
            self.states[indices],
            self.controls[indices],
            self.next_states[indices],
            tuple(self.ids[i] for i in indices),
            self.timestep,
            self.domain,
            None if self.stages is None else self.stages[indices],
        )


def concatenate(batches):
    batches = list(batches)
    if not batches or len({(b.timestep, b.domain) for b in batches}) != 1:
        raise ValueError("concatenation requires batches with one common timestep")
    return Transitions(
        *(
            np.concatenate([getattr(b, name) for b in batches])
            for name in ("states", "controls", "next_states")
        ),
        tuple(identifier for b in batches for identifier in b.ids),
        batches[0].timestep,
        batches[0].domain,
        np.concatenate(
            [b.stages if b.stages is not None else np.zeros(len(b), dtype=int) for b in batches]
        ),
    )
