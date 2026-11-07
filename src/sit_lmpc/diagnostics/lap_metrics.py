"""Per-attempt measured boundary, controller penalty and selected-lambda summaries."""

import numpy as np


class LambdaStatistics:
    """Constant-memory statistics over applied LMPPI controls, by active term."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.steps = 0
        self._terms = {}

    def observe(self, lambdas, terms):
        for name, value in zip(terms, lambdas, strict=True):
            value = float(value)
            total, maximum, count = self._terms.get(name, (0., value, 0))
            self._terms[name] = total + value, max(maximum, value), count + 1
        self.steps += 1

    def summary(self):
        return dict(steps=self.steps, terms={
            name: dict(mean=total / count, max=maximum, steps=count)
            for name, (total, maximum, count) in self._terms.items()
        })


def measured_boundary_metrics(lap, half_width):
    lateral = np.abs(np.asarray(lap.frenet[:, 1], dtype=float))
    finite = np.isfinite(lateral)
    excess = np.maximum(lateral[finite] - half_width, 0)
    return {
        "actual_boundary_max": float(excess.max()) if len(excess) else None,
        "actual_boundary_mean": float(excess.mean()) if len(excess) else None,
        "out_of_bounds_steps": int(np.count_nonzero(excess > 0)),
        "max_abs_lateral_offset": float(lateral[finite].max()) if finite.any() else None,
        "nonfinite_lateral_steps": int(np.count_nonzero(~finite)),
    }


def summarize_lap(lap, number, half_width, violations, *, completed, termination_reason):
    violations = np.asarray(violations).reshape(-1, 2)
    speeds = np.asarray(lap.cartesian[:, 3])
    speeds = speeds[np.isfinite(speeds)]
    return {
        "lap": number,
        "completed": completed,
        "termination_reason": termination_reason,
        "steps": len(lap.times),
        "duration": lap.laptime if len(lap.times) else 0.0,
        "lap_time": lap.laptime if completed and len(lap.times) else None,
        "mean_speed": float(speeds.mean()) if len(speeds) else None,
        # Preserve the legacy fields: these are predicted controller penalties,
        # not measured boundary excess and not a failure probability.
        "safe_set_max": float(violations[:, 0].max()) if len(violations) else None,
        "safe_set_mean": float(violations[:, 0].mean()) if len(violations) else None,
        "boundary_max": float(violations[:, 1].max()) if len(violations) else None,
        "boundary_mean": float(violations[:, 1].mean()) if len(violations) else None,
        **measured_boundary_metrics(lap, half_width),
    }
