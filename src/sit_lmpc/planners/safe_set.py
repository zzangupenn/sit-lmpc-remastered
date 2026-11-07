"""Lap-based safe sets and complete, padded hull constraints."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
import time
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull, QhullError


@struct.dataclass
class Hull:
    equations: jax.Array
    mask: jax.Array


@jax.jit
def hull_distance(states, hull):
    # GPU default matmul may truncate float32 inputs to TF32. Geometry gates
    # need full input precision near the relaxation threshold; this is still
    # float32 when configured, not a switch to float64 or different hull buckets.
    distances = jnp.matmul(
        states, hull.equations[:, :-1].T, precision=jax.lax.Precision.HIGHEST
    ) + hull.equations[:, -1]
    distances = jnp.where(hull.mask, distances, -jnp.inf)
    return jnp.maximum(jnp.max(distances, axis=-1), 0.0)


@partial(jax.jit, static_argnames=("block_size",))
def blocked_hull_distance(states, hull, block_size):
    """One dispatch for all particle queries, with bounded matrix workspace."""
    size = len(states)
    padding = (-size) % block_size
    blocks = jnp.pad(states, ((0, padding), (0, 0))).reshape(-1, block_size, 5)
    distances = jax.lax.map(lambda block: hull_distance(block, hull), blocks)
    return distances.reshape(-1)[:size]


def build_hull(points, bucket=10, dtype=None):
    if len(points) < points.shape[1] + 1:
        raise ValueError("too few points for safe-set hull")
    try:
        equations = ConvexHull(points).equations
    except QhullError as error:
        raise ValueError("safe-set points are degenerate; cannot construct a full hull") from error
    count = len(equations)
    padded = ((count + bucket - 1) // bucket) * bucket
    array = np.zeros((padded, equations.shape[1]), dtype=equations.dtype)
    array[:count] = equations
    # Build the mask on the host: JAX arange/comparison would compile separately
    # for each facet count while the car is running.
    mask = np.arange(padded) < count
    return Hull(jnp.asarray(array, dtype=dtype), jnp.asarray(mask))


def build_safe_geometry(points, config, dtype=None):
    if config.safe_set_query == "implicit":
        from .implicit_hull import build_point_hull
        return build_point_hull(points, config.implicit_hull_point_bucket, dtype)
    if config.safe_set_query == "nf_density":
        raise ValueError("nf_density requires a fitted density snapshot, not a geometric hull")
    return build_hull(points, config.hull_bucket, dtype)


class PrecompiledHullEvaluator:
    """A bounded table of AOT hull scorers, following the original TWIT method."""

    def __init__(self, samples, candidates, bucket=10, limit=1200, dtype=jnp.float64):
        candidate_counts = (candidates,) if isinstance(candidates, int) else tuple(candidates)
        if not candidate_counts or min(samples, *candidate_counts, bucket) < 1 or limit <= bucket:
            raise ValueError("Hull precompile counts must be positive and limit must exceed bucket")
        self.batch_sizes = tuple(sorted({samples, *candidate_counts}))
        self.sizes = tuple(range(bucket, limit, bucket))
        self.bucket, self.dtype = bucket, np.dtype(dtype)
        self.compiled = {}
        self.compiled_many = {}
        self._compiled_batch_sizes = ()

    def precompile(self, emit=lambda *args, **kwargs: None):
        if self._compiled_batch_sizes == self.batch_sizes:
            return
        missing = [(count, size) for count in self.batch_sizes for size in self.sizes
                   if (count, size) not in self.compiled]
        if not missing:
            return
        started = time.perf_counter()
        emit(
            "hull_precompile_started",
            bucket=self.bucket,
            max_facets=self.sizes[-1],
            variants=len(missing),
            dtype=self.dtype.name,
        )
        queries = {count: jnp.zeros((count, 5), dtype=self.dtype) for count in self.batch_sizes}
        hulls = {
            size: Hull(jnp.zeros((size, 6), dtype=self.dtype), jnp.ones(size, dtype=bool))
            for size in self.sizes
        }
        # As in TWIT: lower the finite sizes, compile in parallel, retain executable
        # objects, then execute/wait once so startup includes GPU kernel loading.
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for count, size in missing:
                lowered = hull_distance.lower(queries[count], hulls[size])
                futures[count, size] = pool.submit(lowered.compile)
            compiled = {key: future.result() for key, future in futures.items()}
        for (count, size), executable in compiled.items():
            executable(queries[count], hulls[size]).block_until_ready()
        self.compiled.update(compiled)
        self._compiled_batch_sizes = self.batch_sizes
        emit(
            "hull_precompile_finished",
            variants=len(compiled),
            elapsed=time.perf_counter() - started,
        )

    def ensure_batch(self, count, emit=lambda *args, **kwargs: None):
        """Add an actually active proposal shape, retaining all existing executables."""
        if count not in self.batch_sizes:
            self.batch_sizes = tuple(sorted((*self.batch_sizes, count)))
        self.precompile(emit)

    def evaluate_many(self, states, hull):
        """Batch queries inside a bounded executable, retaining every facet."""
        if states.ndim != 2 or states.shape[1] != 5 or not len(states):
            raise ValueError("particle hull queries must have nonempty shape (N, 5)")
        if not len(hull.mask) or len(hull.mask) % self.bucket:
            raise ValueError("Build the hull using the configured precompile bucket")
        self.precompile_many((len(states),))
        result = None
        maximum = self.sizes[-1]
        for start in range(0, len(hull.mask), maximum):
            end = min(start + maximum, len(hull.mask))
            block = Hull(hull.equations[start:end], hull.mask[start:end])
            distance = self.compiled_many[len(states), end - start](states, block)
            result = distance if result is None else jnp.maximum(result, distance)
        return result

    def precompile_many(self, query_counts, emit=lambda *a, **k: None):
        """Transform before AOT compilation; never nest stored Compiled objects."""
        missing = [(count, size) for count in sorted(set(query_counts)) for size in self.sizes
                   if (count, size) not in self.compiled_many]
        if not missing:
            return
        started = time.perf_counter()
        emit("particle_hull_precompile_started", variants=len(missing))
        queries = {count: jnp.zeros((count, 5), dtype=self.dtype) for count in {n for n, _ in missing}}
        hulls = {size: Hull(jnp.zeros((size, 6), dtype=self.dtype), jnp.ones(size, bool))
                 for size in {s for _, s in missing}}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {}
            for count, size in missing:
                block_size = min(count, max(self.batch_sizes))
                lowered = blocked_hull_distance.lower(queries[count], hulls[size], block_size)
                futures[count, size] = pool.submit(lowered.compile)
            compiled = {key: future.result() for key, future in futures.items()}
        # Warm each executable before driving, including GPU module loading.
        for (count, size), executable in compiled.items():
            executable(queries[count], hulls[size]).block_until_ready()
        self.compiled_many.update(compiled)
        emit("particle_hull_precompile_finished", variants=len(compiled),
             seconds=time.perf_counter() - started)

    def evaluate(self, states, hull):
        if not self.compiled:
            raise RuntimeError("Precompile hull evaluators before control updates")
        if states.ndim != 2 or states.shape[1] != 5 or len(states) not in self.batch_sizes:
            raise ValueError("Hull query shape must match the sample or candidate batch")
        if hull.equations.shape != (len(hull.mask), 6) or not len(hull.mask):
            raise ValueError("Expected nonempty five-dimensional hull equations and mask")
        if len(hull.mask) % self.bucket:
            raise ValueError("Build the hull using the configured precompile bucket")
        maximum = self.sizes[-1]
        # The common case dispatches the exact precompiled size with no new JIT.
        if len(hull.mask) <= maximum:
            return self.compiled[len(states), len(hull.mask)](states, hull)
        # Oversized hulls reuse the same bounded executables; never drop facets or
        # add a new hull scorer to the table during driving.
        result = None
        for start in range(0, len(hull.mask), maximum):
            end = min(start + maximum, len(hull.mask))
            block = Hull(hull.equations[start:end], hull.mask[start:end])
            distance = self.compiled[len(states), end - start](states, block)
            result = distance if result is None else jnp.maximum(result, distance)
        return result


class SafeSet:
    def __init__(self, planner, timestep, track_length, seed=0, randomness=None):
        self.config = planner
        self.timestep = timestep
        self.track_length = track_length
        self.rng = np.random.default_rng(seed) if randomness is None else randomness.host
        self.previous = None
        self.previous_trajectory = None
        self.frenet, self.cartesian = [], []
        self.controls, self.control_masks = [], []
        self.state_ids = []
        self._array_cache = None

    @staticmethod
    def _state_ids(trajectory, count):
        # A learning-lap row is the POST-state of this recorded control step.
        if trajectory is None:
            return np.full((count, 2), -1, dtype=np.int64)
        return np.column_stack((np.full(count, trajectory.attempt_id), trajectory.steps))

    def restore_state_ids(self, trajectories):
        """Recover provenance for checkpoint records from original successes.

        Old checkpoints/imported init arrays do not carry state identities. Only
        exact matches to reconstructed records acquire IDs; unknown imports stay
        ineligible for density calibration until replaced by fresh records.
        """
        replay = SafeSet(self.config, self.timestep, self.track_length)
        for trajectory in trajectories:
            replay.add_lap(trajectory.as_lap(), trajectory)
        self.state_ids = []
        cursor = 0
        for record, cartesian in zip(self.frenet, self.cartesian, strict=True):
            match = None
            # Repeated laps can have identical Frenet arrays. Consume matches
            # in chronological order so each record keeps its own measured IDs.
            for index in range(cursor, len(replay.frenet)):
                if (np.array_equal(record, replay.frenet[index])
                        and np.array_equal(cartesian, replay.cartesian[index])):
                    match, cursor = replay.state_ids[index], index + 1
                    break
            self.state_ids.append(
                np.full((len(record), 2), -1, dtype=np.int64) if match is None else match
            )

    def density_data(self):
        """The retained geometry's unique post-states and measured identities."""
        if not self.frenet or len(self.state_ids) != len(self.frenet):
            return None
        # -1 marks unknown provenance; imported warm-up attempts use a separate
        # negative namespace so they cannot collide with new positive attempts.
        if any(ids.shape != (len(record), 2) or np.any(ids[:, 0] == -1) or np.any(ids[:, 1] < 0)
               for record, ids in zip(self.frenet, self.state_ids, strict=True)):
            return None
        ids = np.concatenate(self.state_ids)
        # Prefixes appear both as an extension and in the next retained lap.
        # Deduplicate by identity, not coordinates; different laps can coincide.
        _, rows = np.unique(ids, axis=0, return_index=True)
        rows.sort()
        return self.arrays()[0][rows, :5], ids[rows]

    def _state_count(self, count):
        if not count:
            return 0
        records, identities = self.frenet[-count:], self.state_ids[-count:]
        if len(identities) == len(records) and all(
            ids.shape == (len(record), 2) and np.all(ids[:, 0] != -1) and np.all(ids[:, 1] >= 0)
            for record, ids in zip(records, identities, strict=True)
        ):
            return len(np.unique(np.concatenate(identities), axis=0))
        # Legacy/state-only init arrays lack measured identities. Cartesian
        # extension copies are exact duplicates even when Frenet progress is lifted.
        return len(np.unique(np.concatenate(self.cartesian[-count:])[:, :7], axis=0))

    @property
    def state_count(self):
        return self._state_count(len(self.frenet))

    def trim(self):
        """Keep the newest sufficient suffix, preserving the configured lap floor."""
        count = min(self.config.safe_set_size, len(self.frenet))
        minimum = self.config.safe_set_min_states
        if minimum:
            while count < len(self.frenet) and self._state_count(count) < minimum:
                count += 1
        for field in ("frenet", "cartesian", "controls", "control_masks", "state_ids"):
            values = getattr(self, field)
            setattr(self, field, values[-count:] if count else [])
        return count

    @property
    def signature(self):
        # Records are immutable; keep ownership in the cache to prevent id reuse.
        return tuple(id(row) for row in self.frenet), tuple(id(row) for row in self.cartesian)

    def arrays(self):
        signature = self.signature
        if self._array_cache is None or self._array_cache[0] != signature:
            self._array_cache = (signature, np.concatenate(self.frenet),
                                 np.concatenate(self.cartesian),
                                 tuple(self.frenet), tuple(self.cartesian))
        return self._array_cache[1:3]

    def window(self, bounds, speed):
        return (bounds[0] - speed * (self.config.reference_interval - 1) * self.timestep,
                bounds[1] + speed * (self.config.reference_interval + 1) * self.timestep)

    def select_bounds(self, bounds, speed):
        return self.select_window(*self.window(bounds, speed))

    def select_window(self, lo, hi, rng=None):
        rng = self.rng if rng is None else rng
        arr_f, arr_c = self.arrays()
        if len(arr_f) > self.config.safe_set_max_points:
            inds = rng.choice(len(arr_f), self.config.safe_set_max_points, replace=False)
            arr_f, arr_c = arr_f[inds], arr_c[inds]
        indices = np.flatnonzero((arr_f[:, 0] > lo) & (arr_f[:, 0] < hi))
        if len(indices) > self.config.hull_selection_points:
            indices = rng.choice(indices, self.config.hull_selection_points, replace=False)
        if len(indices) < 6:
            raise ValueError("not enough local safe-set points for learned control")
        return arr_f[indices], arr_c[indices]

    def speculate(self, window, dtype):
        """Do not consume the real host stream unless this geometry is accepted."""
        rng = deepcopy(self.rng)
        selected, reference = self.select_window(*window, rng=rng)
        hull = build_safe_geometry(selected[:, :5], self.config, dtype=dtype)
        return selected, reference, hull, deepcopy(rng.bit_generator.state), self.signature

    def add_lap(self, lap, trajectory=None):
        # Summaries/dynamics retain measured wrapped states; only this learning
        # view lifts the finish-crossing endpoint to L+s. Offline lap-only input
        # has no collector progress, so unwrap its consecutive post-step states.
        lap = (
            trajectory.as_learning_lap(self.track_length)
            if trajectory is not None
            else lap.with_unwrapped_progress(self.track_length)
        )
        new = False
        if self.previous is not None:
            previous = self.previous
            extend_s = (
                np.max(lap.frenet[:, 3])
                * self.config.extension_seconds
                * self.config.horizon
                * self.timestep
            )
            # Extension is a contiguous STARTING prefix, never a later return
            # to small s or the terminal finish crossing (even for large limits).
            prefix = lap.frenet[:, 0] < min(extend_s, self.track_length)
            indices = np.flatnonzero(np.logical_and.accumulate(prefix))
            old_f = np.column_stack([previous.frenet, previous.times])
            next_f = np.column_stack([lap.frenet[indices], lap.times[indices]])
            next_f[:, 0] += self.track_length
            combined_f = np.concatenate([old_f, next_f])
            combined_f[: len(old_f), -1] -= combined_f[:, -1].max()
            combined_c = np.concatenate([previous.cartesian, lap.cartesian[indices]])
            combined_c = np.column_stack([combined_c, combined_f[:, -1]])
            combined_f.setflags(write=False)
            combined_c.setflags(write=False)
            self.frenet.append(combined_f)
            self.cartesian.append(combined_c)
            ids = np.concatenate((
                self._state_ids(self.previous_trajectory, len(previous.times)),
                self._state_ids(trajectory, len(lap.times))[indices],
            ))
            ids.setflags(write=False)
            while len(self.state_ids) < len(self.frenet) - 1:
                self.state_ids.append(np.full(
                    (len(self.frenet[len(self.state_ids)]), 2), -1, dtype=np.int64
                ))
            self.state_ids.append(ids)
            old_u = np.zeros((len(previous.times), 2))
            old_mask = np.zeros(len(previous.times), dtype=bool)
            if self.previous_trajectory is not None:
                old_u[:-1] = self.previous_trajectory.controls[1:]
                old_mask[:-1] = True
                if trajectory is not None and len(indices) and indices[0] == 0 and np.array_equal(
                    previous.cartesian[-1], trajectory.cartesian[0]
                ):
                    old_u[-1] = trajectory.controls[0]
                    old_mask[-1] = True
            next_u = np.zeros((len(indices), 2))
            next_mask = np.zeros(len(indices), dtype=bool)
            if trajectory is not None:
                next_mask = indices + 1 < len(trajectory.controls)
                next_mask[:-1] &= np.diff(indices) == 1
                next_u[next_mask] = trajectory.controls[indices[next_mask] + 1]
            controls = np.concatenate((old_u, next_u))
            mask = np.concatenate((old_mask, next_mask))
            controls.setflags(write=False)
            mask.setflags(write=False)
            # State-only quick-load records can lack action arrays. Keep explicit
            # invalid placeholders rather than assigning fictional next actions.
            while len(self.controls) < len(self.frenet) - 1:
                record = self.frenet[len(self.controls)]
                self.controls.append(np.zeros((len(record), 2)))
                self.control_masks.append(np.zeros(len(record), dtype=bool))
            self.controls.append(controls)
            self.control_masks.append(mask)
            self.trim()
            new = True
        self.previous = lap
        self.previous_trajectory = trajectory
        return new

    @property
    def ready(self):
        return len(self.frenet) >= self.config.safe_set_size

    def training_data(self, input_mode="state"):
        if not self.frenet:
            raise ValueError("safe set has no completed pairs of laps")
        if input_mode == "state_action":
            if len(self.controls) != len(self.frenet):
                raise ValueError(
                    "state-action value training needs aligned controls; collect fresh warm-up laps"
                )
            rows = [
                np.column_stack((record[mask, :5], control[mask], record[mask, -1]))
                for record, control, mask in zip(
                    self.frenet, self.controls, self.control_masks, strict=True
                )
            ]
            data = np.concatenate(rows)
            if not len(data):
                raise ValueError("state-action value training has no known next actions")
            return data
        if input_mode != "state":
            raise ValueError("input_mode must be state or state_action")
        return np.concatenate(
            [np.column_stack([record[:, :5], record[:, -1]]) for record in self.frenet]
        )

    def select(self, terminals, speed):
        return self.select_bounds((np.min(terminals[:, 0]), np.max(terminals[:, 0])), speed)
