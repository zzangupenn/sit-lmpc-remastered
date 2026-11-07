"""Point-defined hull queries: feasible Frank-Wolfe witnesses and separating bounds.

No facet enumeration or learned classification. Upper distances come from convex
combinations; lower distances come from supporting planes. Unresolved queries
are never called inside. Bounds use ordinary floating-point arithmetic with a
roundoff guard, not an interval-arithmetic proof or a physical safety certificate.
"""

from functools import partial
import time

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class PointHull:
    points: jax.Array
    mask: jax.Array
    center: jax.Array


@struct.dataclass
class DistanceBounds:
    lower: jax.Array
    upper: jax.Array
    iterations: jax.Array
    decided: jax.Array
    converged: jax.Array
    valid: jax.Array


def build_point_hull(points, bucket=64, dtype=None):
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2 or not len(array) or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError("point hull needs nonempty finite (N, D) points")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if type(bucket) is not int or bucket < 1:
        raise ValueError("point bucket must be a positive integer")
    # Center on the host to avoid cancellation from unwrapped lap progress.
    center = array.mean(axis=0, dtype=np.float64).astype(array.dtype)
    count = ((len(array) + bucket - 1) // bucket) * bucket
    padded = np.zeros((count, array.shape[1]), dtype=array.dtype)
    padded[: len(array)] = array - center
    return PointHull(
        jnp.asarray(padded), jnp.asarray(np.arange(count) < len(array)), jnp.asarray(center)
    )


def _block_bounds(states, hull, relaxation, *, max_iterations, tolerance, membership_only):
    points, mask = hull.points, hull.mask
    valid = jnp.all(jnp.isfinite(states), axis=-1)
    x = jnp.where(valid[:, None], states - hull.center, 0.0)
    point_norms = jnp.sum(points * points, axis=-1)
    squared = (
        jnp.sum(x * x, axis=-1)[:, None]
        + point_norms[None]
        - 2 * jnp.matmul(x, points.T, precision=jax.lax.Precision.HIGHEST)
    )
    nearest = points[jnp.argmin(jnp.where(mask[None], squared, jnp.inf), axis=-1)]
    # Both the centroid and each vertex are feasible witnesses.
    q = jnp.where(
        (jnp.sum((x - nearest) ** 2, axis=-1) < jnp.sum(x * x, axis=-1))[:, None],
        nearest,
        jnp.zeros_like(x),
    )
    magnitude = 1 + jnp.linalg.norm(x, axis=-1) + jnp.sqrt(jnp.max(jnp.where(mask, point_norms, 0)))
    guard = 64 * jnp.finfo(x.dtype).eps * magnitude

    def bounds(q, old_lower):
        r = x - q
        norm = jnp.linalg.norm(r, axis=-1)
        direction = r / jnp.maximum(norm[:, None], jnp.finfo(x.dtype).tiny)
        projections = jnp.matmul(direction, points.T, precision=jax.lax.Precision.HIGHEST)
        index = jnp.argmax(jnp.where(mask[None], projections, -jnp.inf), axis=-1)
        support = points[index]
        lower = jnp.maximum(
            0, jnp.sum(direction * x, axis=-1) - jnp.sum(direction * support, axis=-1) - guard
        )
        upper = norm + guard
        lower = jnp.minimum(jnp.maximum(old_lower, lower), upper)
        decided = (upper <= relaxation) | (lower > relaxation)
        converged = upper - lower <= jnp.maximum(tolerance, 2 * guard)
        done = decided if membership_only else (converged | (upper <= relaxation))
        return lower, upper, support, decided, converged, done | ~valid

    lower, upper, support, decided, converged, done = bounds(q, jnp.zeros(len(x), dtype=x.dtype))
    initial = (
        jnp.asarray(0),
        q,
        lower,
        upper,
        support,
        decided,
        converged,
        done,
        jnp.zeros(len(x), dtype=jnp.int32),
    )

    def condition(carry):
        return (carry[0] < max_iterations) & jnp.any(~carry[7])

    def step(carry):
        i, q, lower, upper, support, decided, converged, done, iterations = carry
        delta = support - q
        numerator = jnp.sum((x - q) * delta, axis=-1)
        gamma = jnp.clip(
            numerator / jnp.maximum(jnp.sum(delta * delta, axis=-1), jnp.finfo(x.dtype).tiny), 0, 1
        )
        proposed = q + gamma[:, None] * delta
        improves = jnp.sum((x - proposed) ** 2, axis=-1) <= jnp.sum((x - q) ** 2, axis=-1)
        q = jnp.where((~done & improves)[:, None], proposed, q)
        lower, upper, support, decided, converged, new_done = bounds(q, lower)
        return (
            i + 1,
            q,
            lower,
            upper,
            support,
            decided,
            converged,
            done | new_done,
            iterations + (~done).astype(jnp.int32),
        )

    _, _, lower, upper, _, decided, converged, _, iterations = jax.lax.while_loop(
        condition, step, initial
    )
    return DistanceBounds(
        jnp.where(valid, lower, 0),
        jnp.where(valid, upper, jnp.inf),
        iterations,
        decided & valid,
        converged & valid,
        valid,
    )


@partial(jax.jit, static_argnames=("max_iterations", "tolerance", "membership_only", "block_size"))
def implicit_hull_bounds(
    states,
    hull,
    relaxation,
    *,
    max_iterations=128,
    tolerance=1e-4,
    membership_only=False,
    block_size=1024,
):
    """Return conservative distance bounds in the supplied (unscaled) coordinates.

    Inside means upper<=relaxation, outside means lower>relaxation. A remaining
    gap straddling the threshold is unresolved; callers must not accept it.
    Unlike max-facet violation, this is a Euclidean hull-offset tolerance.
    """
    size = len(states)
    block = min(size, block_size)
    padded = jnp.pad(states, ((0, (-size) % block), (0, 0))).reshape(-1, block, states.shape[-1])
    result = jax.lax.map(
        lambda x: _block_bounds(
            x,
            hull,
            relaxation,
            max_iterations=max_iterations,
            tolerance=tolerance,
            membership_only=membership_only,
        ),
        padded,
    )
    return jax.tree.map(lambda x: x.reshape(-1)[:size], result)


@jax.jit
def bound_statistics(result):
    return dict(
        queries=jnp.asarray(len(result.upper)),
        unresolved=jnp.sum(result.valid & ~result.decided),
        invalid=jnp.sum(~result.valid),
        iterations=jnp.sum(result.iterations),
        max_iterations=jnp.max(result.iterations),
        max_gap=jnp.max(jnp.where(result.valid, result.upper - result.lower, 0)),
    )


class ImplicitHullEvaluator:
    """Bounded, upfront-compiled point buckets for nominal and scenario queries."""

    def __init__(self, config, candidates, dtype):
        self.config, self.dtype = config, np.dtype(dtype)
        self.batch_sizes = tuple(sorted({config.samples, candidates}))
        bucket = config.implicit_hull_point_bucket
        self.sizes = tuple(
            range(
                bucket, ((config.hull_selection_points + bucket - 1) // bucket) * bucket + 1, bucket
            )
        )
        self.compiled, self.compiled_many = {}, {}
        self.statistics = []

    def _compile(self, counts, membership, emit):
        table = self.compiled_many if membership else self.compiled
        missing = [
            (count, size)
            for count in sorted(set(counts))
            for size in self.sizes
            if (count, size) not in table
        ]
        if not missing:
            return
        start = time.perf_counter()
        emit("implicit_hull_precompile_started", variants=len(missing), membership_only=membership)
        for count, size in missing:
            hull = PointHull(
                jnp.zeros((size, 5), self.dtype), jnp.ones(size, bool), jnp.zeros(5, self.dtype)
            )
            query = jnp.zeros((count, 5), self.dtype)
            executable = implicit_hull_bounds.lower(
                query,
                hull,
                self.config.safe_set_relaxation,
                max_iterations=self.config.implicit_hull_max_iterations,
                tolerance=self.config.implicit_hull_tolerance,
                membership_only=membership,
                block_size=self.config.implicit_hull_block_size,
            ).compile()
            jax.block_until_ready(executable(query, hull, self.config.safe_set_relaxation))
            table[count, size] = executable
        emit(
            "implicit_hull_precompile_finished",
            variants=len(missing),
            membership_only=membership,
            elapsed=time.perf_counter() - start,
        )

    def precompile(self, emit=lambda *a, **k: None):
        self._compile(self.batch_sizes, False, emit)

    def ensure_batch(self, count, emit=lambda *a, **k: None):
        if count not in self.batch_sizes:
            self.batch_sizes = tuple(sorted((*self.batch_sizes, count)))
        self.precompile(emit)

    def precompile_many(self, counts, emit=lambda *a, **k: None):
        self._compile(counts, True, emit)

    def reset_statistics(self):
        self.statistics.clear()

    def _evaluate(self, states, hull, membership):
        table = self.compiled_many if membership else self.compiled
        key = len(states), len(hull.mask)
        if self.config.hull_precompile:
            if key not in table:
                raise RuntimeError("implicit hull query shape was not precompiled")
            result = table[key](states, hull, self.config.safe_set_relaxation)
        else:
            result = implicit_hull_bounds(
                states,
                hull,
                self.config.safe_set_relaxation,
                max_iterations=self.config.implicit_hull_max_iterations,
                tolerance=self.config.implicit_hull_tolerance,
                membership_only=membership,
                block_size=self.config.implicit_hull_block_size,
            )
        self.statistics.append(bound_statistics(result))
        # In particular, unresolved upper>relaxation is conservatively outside.
        return result.upper

    def evaluate(self, states, hull):
        return self._evaluate(states, hull, False)

    def evaluate_many(self, states, hull):
        return self._evaluate(states, hull, True)

    def diagnostics(self):
        if not self.statistics:
            return {}
        return {
            "implicit_hull_" + key: (
                jnp.max(jnp.stack([s[key] for s in self.statistics]))
                if key.startswith("max_")
                else jnp.sum(jnp.stack([s[key] for s in self.statistics]))
            )
            for key in self.statistics[0]
        }
