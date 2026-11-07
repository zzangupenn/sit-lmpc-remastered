"""Calibrated conditional safe-trajectory density, separate from value and risk.

Quantile uses all successful states; sigma uses the historical 90/10 split. Retention is not
P(safe | accepted), an invariant set, or a conformal guarantee under policy drift.
"""

from dataclasses import asdict, fields
from functools import partial
import hashlib
import json
from pathlib import Path

from flax import serialization, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sit_lmpc.config.schema import normalize_density_settings
from sit_lmpc.data.partitions import state_partition_indices
from sit_lmpc.models.value.flow import NeuralSplineFlow
from sit_lmpc.training.features import inference_snapshot
from sit_lmpc.training.value import train_epoch


@struct.dataclass
class DensitySnapshot:
    state: object
    offset: jax.Array
    scale: jax.Array
    track_length: jax.Array
    cutoff: jax.Array  # Legacy storage name: minimum safe-set NLL, the penalty's zero reference.
    penalty_scale: jax.Array
    generation: int
    penalty_kind: str = struct.field(pytree_node=False, default="linear", kw_only=True)
    penalty_normalizer: jax.Array = struct.field(default=np.log(2.), kw_only=True)
    relaxation_cutoff: jax.Array | None = struct.field(default=None, kw_only=True)

    @property
    def fixed_penalty(self):
        return self.penalty_kind in ("bounded_linear", "log1p")

    @property
    def mask(self):
        # Compatible geometry diagnostics, without pretending there are facets.
        return jnp.zeros(0, dtype=bool)


@struct.dataclass
class LocalDensitySnapshot(DensitySnapshot):
    window_min: jax.Array
    window_max: jax.Array

    @classmethod
    def create(cls, snapshot, lo, hi):
        dtype = snapshot.offset.dtype
        lower, upper = np.asarray(lo, dtype=dtype), np.asarray(hi, dtype=dtype)
        if float(lower) > lo:
            lower = np.nextafter(lower, np.asarray(-np.inf, dtype=dtype))
        if float(upper) < hi:
            upper = np.nextafter(upper, np.asarray(np.inf, dtype=dtype))
        return cls(**{f.name: getattr(snapshot, f.name) for f in fields(DensitySnapshot)},
                   window_min=jnp.asarray(lower), window_max=jnp.asarray(upper))


def initial_state(config, key, *, template=None):
    model = NeuralSplineFlow(
        n_dim=4,
        n_context=2,
        n_transforms=config.layers,
        hidden_dims=[config.hidden_dims, config.hidden_dims],
        n_bins=config.bins,
        activation="swish",
        range_min=-5.0,
        range_max=5.0,
    )
    params = model.init(key, jnp.zeros((1, 4)), jnp.zeros((1, 2)))
    if template is not None:
        # Reuse static callables to keep the compiled training kernels while
        # resetting parameters, optimizer moments, and the step counter.
        return template.replace(step=0, params=params, opt_state=template.tx.init(params))
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(8.0), optax.adam(config.learning_rate)),
    )


def progress_context(states, length):
    phase = 2 * jnp.pi * jnp.remainder(states[:, 0], length) / length
    return jnp.column_stack((jnp.sin(phase), jnp.cos(phase)))


@partial(jax.jit, static_argnames=("block_size",))
def density_scores(snapshot, states, block_size=1024):
    """Physical-coordinate conditional NLL; deterministic, no latent sampling."""
    size = len(states)
    block = min(size, block_size)
    rows = jnp.pad(states[:, :5], ((0, (-size) % block), (0, 0))).reshape(-1, block, 5)

    def score(x):
        valid = jnp.isfinite(x).all(axis=1)
        clean = jnp.where(valid[:, None], x, 0.0)
        targets = (clean[:, 1:5] - snapshot.offset) / snapshot.scale
        context = progress_context(clean, snapshot.track_length)
        nll = -snapshot.state.apply_fn(snapshot.state.params, targets, context)
        nll += jnp.log(snapshot.scale).sum()
        return jnp.where(valid & jnp.isfinite(nll), nll, jnp.inf)

    return jax.lax.map(score, rows).reshape(-1)[:size]


@jax.jit
def density_penalty(snapshot, states):
    scores = density_scores(snapshot, states)
    return _penalty_from_scores(snapshot, states, scores)


def relaxation_cutoff(snapshot):
    """The frozen raw-NLL acceptance bound; null explicitly shares the reward cutoff."""
    return snapshot.cutoff if snapshot.relaxation_cutoff is None else snapshot.relaxation_cutoff


@jax.jit
def density_violation(snapshot, states):
    """Candidate/scenario violation, independent of the sampled reward cutoff.

    Reuse the update's compression width and normalizer, moving only the zero
    boundary. Zero means NLL <= the acceptance bound and local support holds.
    """
    check = snapshot.replace(cutoff=relaxation_cutoff(snapshot))
    return _penalty_from_scores(check, states, density_scores(snapshot, states))


def _log1p_excess(scores, cutoff, scale):
    """log(1 + max(NLL-cutoff, 0)/scale), without overflowing finite inputs."""
    excess = jnp.maximum(scores - cutoff, 0.)
    # Opposite-sign finite scores can overflow the subtraction; the log of
    # their positive difference is the log-sum of their absolute values.
    log_excess = jnp.where(jnp.isfinite(excess), jnp.log(excess),
                          jnp.logaddexp(jnp.log(jnp.abs(scores)), jnp.log(jnp.abs(cutoff))))
    return jnp.logaddexp(jnp.asarray(0., scores.dtype), log_excess - jnp.log(scale))


def _penalty_from_scores(snapshot, states, scores):
    if snapshot.penalty_kind == "log1p":
        penalty = _log1p_excess(scores, snapshot.cutoff, snapshot.penalty_scale)
        penalty /= jnp.asarray(snapshot.penalty_normalizer, scores.dtype)
    else:
        penalty = jnp.clip((scores - snapshot.cutoff) / snapshot.penalty_scale, 0.0, 1.0)
    if isinstance(snapshot, LocalDensitySnapshot):
        # A local cutoff has no support at unrelated track positions. Every
        # proposal/scenario uses this same frozen window, including rerolls.
        supported = (states[:, 0] > snapshot.window_min) & (states[:, 0] < snapshot.window_max)
        unsupported = jnp.maximum(penalty, 1.) if snapshot.penalty_kind == "log1p" else 1.
        penalty = jnp.where(supported, penalty, unsupported)
    return jnp.where(jnp.isfinite(scores), penalty, jnp.inf)


@jax.jit
def sample_normalized_density(snapshot, states, valid=None):
    """Freeze the sampled normalization for all proposals and subsequent rerolls."""
    scores = density_scores(snapshot, states)
    valid = jnp.isfinite(scores) if valid is None else valid & jnp.isfinite(scores)
    maximum = jnp.max(jnp.where(valid, scores, -jnp.inf))
    calibrated_upper = snapshot.cutoff + snapshot.penalty_scale
    if snapshot.penalty_kind == "log1p":
        active = maximum > snapshot.cutoff
        normalizer = _log1p_excess(maximum, snapshot.cutoff, snapshot.penalty_scale)
        # No violating valid samples: keep zeros at zero and a calibrated
        # positive denominator for any later violating rerolled candidates.
        normalizer = jnp.where(active, normalizer, jnp.log(jnp.asarray(2., scores.dtype)))
        frozen = snapshot.replace(penalty_normalizer=normalizer)
        metrics = dict(safe_set_density_cutoff=frozen.cutoff,
                       safe_set_density_penalty_scale=frozen.penalty_scale,
                       safe_set_density_upper_cutoff=jnp.where(active, maximum, calibrated_upper),
                       safe_set_density_calibrated_upper_cutoff=calibrated_upper,
                       safe_set_density_sample_max_nll=maximum,
                       safe_set_density_sample_max_active=active.astype(scores.dtype),
                       safe_set_density_log_normalizer=normalizer)
        return frozen, _penalty_from_scores(frozen, states, scores), metrics
    upper = jnp.minimum(maximum, calibrated_upper)
    # An entirely zero-penalty batch remains zero. Retain the calibrated scale
    # for candidate scoring instead of dividing by zero or assigning a penalty.
    scale = jnp.where(upper > snapshot.cutoff, upper - snapshot.cutoff, snapshot.penalty_scale)
    frozen = snapshot.replace(penalty_scale=scale)
    metrics = dict(safe_set_density_cutoff=frozen.cutoff,
                   safe_set_density_penalty_scale=scale,
                   safe_set_density_upper_cutoff=frozen.cutoff + scale,
                   safe_set_density_calibrated_upper_cutoff=calibrated_upper,
                   safe_set_density_sample_max_nll=maximum)
    return frozen, _penalty_from_scores(frozen, states, scores), metrics


def calibration_bound(scores, quantile, dtype=None):
    """Observed NLL quantile, rounded upward for inference precision."""
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("density calibration requires finite safe-set scores")
    bound = float(np.quantile(scores, quantile, method="higher"))
    return _round_bound(bound, dtype)


def _round_bound(bound, dtype):
    if dtype is not None:
        rounded = np.asarray(bound, dtype=dtype)
        if float(rounded) < bound:
            rounded = np.nextafter(rounded, np.asarray(np.inf, dtype=rounded.dtype))
        bound = float(rounded)
    return bound


def calibrate(scores, target, dtype=None, *, penalty_upperbound_quantile=None, upper_bound=None):
    """Calibrate both NF bounds; omitted upper policy preserves the Walker helper API."""
    cutoff = calibration_bound(scores, target)
    if upper_bound is not None:
        upper = float(upper_bound)
    elif penalty_upperbound_quantile is None:
        # Walker occupancy still uses its existing quantile/IQR policy. Racecar
        # callers always pass their independently configured upper quantile.
        upper = cutoff + max(float(np.quantile(scores, .75) - np.quantile(scores, .25)), 1e-3)
    else:
        upper = calibration_bound(scores, penalty_upperbound_quantile)
    if upper < cutoff:
        raise ValueError(
            f"density upper bound ({upper:g}) is below "
            f"the lower bound ({cutoff:g}); adjust the bound settings"
        )
    # Tied empirical quantiles (including a one-state local window) need a
    # positive interval. Preserve the cutoff and use a small numerical floor.
    if upper == cutoff:
        upper = cutoff + 1e-3
    if dtype is not None:
        cutoff = calibration_bound(scores, target, dtype)
        limit = np.asarray(upper, dtype=dtype)
        if float(limit) < upper:
            limit = np.nextafter(limit, np.asarray(np.inf, dtype=limit.dtype))
        if float(limit) <= cutoff:
            limit = np.nextafter(np.asarray(cutoff, dtype=dtype), np.asarray(np.inf, dtype=limit.dtype))
        spread = float(limit) - cutoff
        spread = float(np.asarray(spread, dtype=dtype))
    else:
        spread = upper - cutoff
    if not np.isfinite([cutoff, spread]).all() or spread <= 0:
        raise ValueError("density calibration bounds must be finite with positive separation")
    return cutoff, spread


def calibrate_config(scores, config, dtype=None):
    """Quantile uses the minimum; sigma restores the sweep's lower/upper bounds."""
    if config.method == "sigma":
        values = np.asarray(scores, dtype=np.float64)
        # Match the sweep's population standard deviation and constant-score floor.
        calibration_bound(values, config.sigma_cutoff_quantile)
        upper = float(values.mean()) + config.penalty_sigma * max(float(values.std(ddof=0)), 1e-3)
        return calibrate(values, config.sigma_cutoff_quantile, dtype, upper_bound=upper)
    return calibrate(scores, 0., dtype, penalty_upperbound_quantile=config.penalty_upperbound_quantile)


def calibrate_relaxation(scores, config, dtype=None):
    quantile = config.sigma_cutoff_quantile if config.method == "sigma" else config.relaxation_quantile
    bound = calibration_bound(scores, quantile, dtype)
    if not np.isfinite(bound):
        raise ValueError("NF relaxation NLL bound must be finite in the inference dtype")
    return bound


def density_penalty_kind(config):
    if config.penalty_transform == "log1p":
        return "log1p"
    return "bounded_linear"


def density_method_metadata(config):
    """Only active-method settings are attached to controller decisions."""
    sigma = config.method == "sigma"
    return dict(method=config.method,
                penalty_origin="quantile_cutoff" if sigma else "safe_set_min",
                relaxation_quantile=(config.sigma_cutoff_quantile if sigma else config.relaxation_quantile),
                penalty_upperbound_quantile=None if sigma else config.penalty_upperbound_quantile,
                **(dict(penalty_sigma=config.penalty_sigma,
                        sigma_cutoff_quantile=config.sigma_cutoff_quantile) if sigma else {}))


class LocalDensityCalibration:
    """Cached scores for every unique state in the retained NF safe-set history.

    All aliases of an extension state can locate it in a selected progress
    window, but a measured identity contributes only one calibration score.
    This host-side selector consumes no RNG and never evaluates the NF.
    """

    def __init__(self, snapshot, validation, safe_set):
        self.snapshot = snapshot
        self.signature = None if safe_set is None else safe_set.signature
        self._records = (validation if safe_set is None else
                         (tuple(safe_set.frenet), tuple(safe_set.cartesian)))
        ids = np.asarray(validation["calibration_ids"])
        scores = np.asarray(validation["calibration_scores"])
        if len(ids) != len(scores) or not np.isfinite(scores).all():
            raise ValueError("local density calibration needs aligned finite safe-set scores")
        lookup = {tuple(key): float(score) for key, score in zip(ids, scores, strict=True)}
        if len(lookup) != len(ids):
            raise ValueError("local density calibration identities must be unique")
        if safe_set is None:
            # Independent NF history can outlive the controller's geometric
            # records. Periodic aliases locate all its states in the same
            # terminal-progress window, with each measured identity counted once.
            progress = np.remainder(validation["calibration_states"][:, 0], float(snapshot.track_length))
            length = float(snapshot.track_length)
            self.progress = np.concatenate((progress - length, progress, progress + length))
            self.scores, self.ids = np.tile(scores, 3), np.tile(ids, (3, 1))
        else:
            source_ids = np.concatenate(safe_set.state_ids)
            source_scores = np.asarray([lookup.get(tuple(key), np.nan) for key in source_ids])
            known = np.isfinite(source_scores)
            self.progress = safe_set.arrays()[0][known, 0].copy()
            self.scores = source_scores[known]
            self.ids = source_ids[known].copy()
        unique, self.identities = np.unique(self.ids, axis=0, return_inverse=True)
        if len(unique) != len(ids):
            raise ValueError("local calibration states do not belong to the retained safe set")
        self.count = len(unique)
        for value in (self.progress, self.scores, self.ids, self.identities):
            value.setflags(write=False)
        self._selection, self._snapshot = None, None

    def select(self, safe_set, bounds, speed, config):
        lo, hi = safe_set.window(bounds, speed)
        metrics = dict(safe_set_density_calibration="local",
                       safe_set_density_window_min=float(lo), safe_set_density_window_max=float(hi))
        if self.signature is not None and self.signature != safe_set.signature:
            return None, {**metrics, "safe_set_density_calibration_status": "stale_safe_set"}
        # The sampled endpoints determine the same window as geometric safe-set
        # selection. Its state count is an outcome, never a target to expand to.
        selected = np.flatnonzero((self.progress > lo) & (self.progress < hi))
        identities, first = np.unique(self.identities[selected], return_index=True)
        if not len(identities):
            return None, {**metrics, "safe_set_density_calibration_status": "no_local_calibration_states",
                          "safe_set_density_calibration_states": 0}
        key = (config.method, config.penalty_upperbound_quantile, config.penalty_transform,
               config.relaxation_quantile, config.sigma_cutoff_quantile, config.penalty_sigma,
               tuple(identities))
        if key != self._selection:
            cutoff, spread = calibrate_config(self.scores[selected[first]], config, self.snapshot.offset.dtype)
            calibrated = self.snapshot.replace(
                cutoff=jnp.asarray(cutoff, self.snapshot.offset.dtype),
                penalty_scale=jnp.asarray(spread, self.snapshot.offset.dtype),
                penalty_kind=density_penalty_kind(config),
                relaxation_cutoff=jnp.asarray(
                    calibrate_relaxation(self.scores[selected[first]], config, self.snapshot.offset.dtype),
                    self.snapshot.offset.dtype),
            )
            self._snapshot = LocalDensitySnapshot.create(calibrated, lo, hi)
            self._selection = key
            self._parameters = cutoff, spread
            self._bounds = lo, hi
        elif self._bounds != (lo, hi):
            self._snapshot = LocalDensitySnapshot.create(self._snapshot, lo, hi)
            self._bounds = lo, hi
        cutoff, spread = self._parameters
        metrics.update(safe_set_density_calibration_status="ready",
                       safe_set_density_calibration_states=len(identities),
                       safe_set_density_window_min=float(lo), safe_set_density_window_max=float(hi),
                       safe_set_density_window_scale=1.0,
                       safe_set_density_cutoff=cutoff, safe_set_density_penalty_scale=spread)
        metrics.update({f"safe_set_density_{key}": value for key, value in density_method_metadata(config).items()})
        metrics["safe_set_density_nll_min"] = (float(np.min(self.scores[selected[first]]))
                                              if config.method == "sigma" else cutoff)
        metrics["safe_set_density_relaxation_cutoff"] = float(relaxation_cutoff(self._snapshot))
        metrics["safe_set_density_upper_cutoff"] = cutoff + spread
        return self._snapshot, metrics


class SafeSetDensityTrainer:
    """Synchronous attempt-boundary fit/calibration and atomic snapshot publication."""

    def __init__(
        self,
        config,
        split_config,
        simulation,
        track_length,
        safe_set_size,
        directory,
        randomness,
        emit=lambda *a, **k: None,
        *,
        safe_set_min_states=0,
        separate_training=False,
    ):
        self.config, self.split_config, self.simulation = config, split_config, simulation
        self.track_length, self.keep = float(track_length), safe_set_size + 1
        self.safe_set_min_states = safe_set_min_states
        self.separate_training = separate_training
        self.safe_set_training = {}
        self.directory, self.randomness, self.emit = Path(directory), randomness, emit
        self._initialization_key = randomness.next_key()
        self.state = initial_state(config, self._initialization_key)
        self.snapshot, self.offset, self.scale = None, None, None
        self.data_signature = None
        self.generation, self.episodes = 0, {}
        self.diagnostics = {"status": "warmup"}
        self.validation = None
        self.local_calibration = None
        self.initial_data = None

    def observe(self, trajectory):
        if trajectory.outcome != "success":
            return
        ids = np.column_stack((np.full(len(trajectory.steps), trajectory.attempt_id),
                               trajectory.steps))
        self.episodes[trajectory.attempt_id] = ids
        if self.separate_training:
            states = trajectory.as_learning_lap(self.track_length).frenet[:, :5].copy()
            self.safe_set_training[trajectory.attempt_id] = (states, ids.copy())
            return
        if self.safe_set_min_states:
            # Prune after the new lap's adaptive geometry has been selected.
            # Pruning here could discard an older lap that must remain active.
            return
        # Imported warm-up identities precede experiment-local attempts. Keep
        # their measured provenance while the imported geometry is being replaced.
        imported = {key: value for key, value in self.episodes.items() if key < -1}
        recent = sorted((key, value) for key, value in self.episodes.items() if key >= 0)
        self.episodes = dict(sorted((*imported.items(), *recent[-self.keep:])))

    def _seed_training_history(self, safe_set):
        data = safe_set.density_data()
        if data is None:
            return set()
        states, ids = data
        for attempt in np.unique(ids[:, 0]):
            selected = ids[:, 0] == attempt
            required_states, required_ids = states[selected], ids[selected]
            if attempt not in self.safe_set_training:
                self.safe_set_training[int(attempt)] = (required_states.copy(), required_ids.copy())
            else:
                retained_states, retained_ids = self.safe_set_training[attempt]
                missing = ~np.isin(required_ids[:, 1], retained_ids[:, 1])
                if missing.any():
                    merged_ids = np.concatenate((retained_ids, required_ids[missing]))
                    merged_states = np.concatenate((retained_states, required_states[missing]))
                    order = np.argsort(merged_ids[:, 1], kind="stable")
                    self.safe_set_training[int(attempt)] = (merged_states[order], merged_ids[order])
            # Identified controller states are measured provenance, including
            # records omitted by older NF checkpoints or partially seeded init.
            previous = self.episodes.get(attempt, np.empty((0, 2), dtype=ids.dtype))
            self.episodes[int(attempt)] = np.unique(np.concatenate((previous, required_ids)), axis=0)
        return set(map(int, np.unique(ids[:, 0])))

    def _training_roles(self):
        roles = {}
        for episode_ids in self.episodes.values():
            for role, rows in enumerate(state_partition_indices(episode_ids, .1, 0., self.randomness.seed)):
                roles.update((tuple(key), role) for key in episode_ids[rows])
        return roles

    def _trim_training_history(self, safe_set):
        required = self._seed_training_history(safe_set)
        # Fixed-step fits retain distinct data independently of their update
        # budget. A small history is reused in further passes, not enlarged to
        # batch_size * steps. Legacy epoch configs keep their original floor.
        fixed_steps = self.config.steps is not None
        target = (self.safe_set_min_states if fixed_steps
                  else self.config.min_training_batches * self.config.batch_size)
        roles = self._training_roles() if self.config.method == "sigma" else None
        retained, count = {}, 0
        for attempt in sorted(self.safe_set_training, reverse=True):
            states, ids = self.safe_set_training[attempt]
            retained[attempt] = (states, ids)
            count += len(ids) if roles is None else sum(roles[tuple(key)] == 0 for key in ids)
            # The controller's extended laps lag the newest complete NF lap.
            # Never discard an attempt still supplying any controller state.
            if len(retained) >= self.keep - 1 and count >= target and required <= retained.keys():
                break
        self.safe_set_training = dict(sorted(retained.items()))
        self.episodes = {key: self.episodes[key] for key in self.safe_set_training}

    def training_history(self):
        """Numeric, portable NF history, including original measured identities."""
        if not self.safe_set_training:
            return None
        records = [self.safe_set_training[key] for key in sorted(self.safe_set_training)]
        return dict(states=np.concatenate([item[0] for item in records]),
                    ids=np.concatenate([item[1] for item in records]),
                    episode_ids=np.concatenate([self.episodes[key] for key in sorted(self.safe_set_training)]))

    def _load_training_history(self, history, *, imported=False):
        states = np.asarray(history["states"]).copy()
        ids = np.asarray(history["ids"]).copy()
        episode_ids = np.asarray(history["episode_ids"]).copy()
        if (states.shape != (len(ids), 5) or ids.shape != (len(ids), 2)
                or episode_ids.ndim != 2 or episode_ids.shape[1] != 2
                or not np.isfinite(states).all()
                or any(not np.issubdtype(value.dtype, np.integer) for value in (ids, episode_ids))
                or any(len(np.unique(value, axis=0)) != len(value) for value in (ids, episode_ids))
                or not set(map(tuple, ids)) <= set(map(tuple, episode_ids))):
            raise ValueError("NF training history must contain unique aligned measured states and identities")
        if imported:
            for value in (ids, episode_ids):
                if np.any(value[:, 0] < 1) or np.any(value[:, 0] >= 2**32 - 1):
                    raise ValueError("NF initial training history requires positive original attempt identities")
                value[:, 0] -= 2**32
        for attempt in np.unique(ids[:, 0]):
            self.safe_set_training[int(attempt)] = (states[ids[:, 0] == attempt], ids[ids[:, 0] == attempt])
            self.episodes[int(attempt)] = episode_ids[episode_ids[:, 0] == attempt]
        self.episodes = dict(sorted(self.episodes.items()))

    def restore_training_history(self, checkpoint):
        path = Path(checkpoint) / "safe_set_density" / "training_history.npz"
        if self.separate_training and path.is_file():
            with np.load(path, allow_pickle=False) as data:
                self._load_training_history(data)

    def import_initial(self, initial, safe_set):
        """Recover NF provenance only from original measured post-states.

        Schema-3 init files carry full successful warm-up transitions. Extended
        safe-set rows only acquire an identity when their Cartesian state is an
        exact, unambiguous match to one of those measured post-states.
        """
        batch = initial.transitions
        if batch is None or not len(batch):
            return False
        if not np.issubdtype(np.asarray(batch.ids).dtype, np.integer):
            raise ValueError("NF initial transitions require integer identities")
        ids = np.asarray(batch.ids, dtype=np.int64).copy()
        if np.any(ids[:, 0] < 1) or np.any(ids[:, 0] >= 2**32 - 1) or np.any(ids[:, 1] < 0):
            raise ValueError("NF initial transitions require positive original attempt identities")
        # A separate namespace prevents collisions with new attempt 1, 2, ...;
        # modulo 2**32 recovers each source attempt's original identity.
        ids[:, 0] -= 2**32
        aligned = self._import_initial_data(batch.next_states, ids, safe_set)
        history = getattr(initial, "density_training", None)
        if self.separate_training and history is not None:
            self._load_training_history(history, imported=True)
        return aligned

    def restore_initial(self, checkpoint, safe_set):
        path = Path(checkpoint) / "safe_set_density" / "initial.npz"
        if not path.is_file():
            return False
        with np.load(path, allow_pickle=False) as data:
            return self._import_initial_data(data["cartesian"], data["ids"], safe_set)

    def _import_initial_data(self, cartesian, ids, safe_set):
        cartesian, ids = np.asarray(cartesian).copy(), np.asarray(ids).copy()
        if (cartesian.shape != (len(ids), 7) or ids.shape != (len(ids), 2)
                or not np.isfinite(cartesian).all() or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids[:, 0] >= -1) or np.any(ids[:, 1] < 0)
                or len(np.unique(ids, axis=0)) != len(ids)):
            raise ValueError("NF initial post-states and identities are invalid")
        lookup = {}
        for point, identity in zip(cartesian, ids, strict=True):
            key = tuple(point)
            value = tuple(identity)
            lookup[key] = value if key not in lookup or lookup[key] == value else None
        restored = []
        known_records = safe_set.state_ids or [np.full((len(record), 2), -1, dtype=np.int64)
                                               for record in safe_set.cartesian]
        for record, known in zip(safe_set.cartesian, known_records, strict=True):
            identities = known.copy()
            for index in np.flatnonzero(identities[:, 0] == -1):
                identity = lookup.get(tuple(record[index, :7]))
                if identity is not None:
                    identities[index] = identity
            restored.append(identities)
        safe_set.state_ids = restored
        if self.safe_set_min_states:
            safe_set.trim()
        for attempt in np.unique(ids[:, 0]):
            self.episodes[int(attempt)] = ids[ids[:, 0] == attempt]
        self.episodes = dict(sorted(self.episodes.items()))
        if self.separate_training:
            self._seed_training_history(safe_set)
        self.initial_data = cartesian, ids
        root = self.directory / "safe_set_density"
        root.mkdir(parents=True, exist_ok=True)
        pending = root / ".initial-pending.npz"
        np.savez_compressed(pending, cartesian=cartesian, ids=ids)
        pending.replace(root / "initial.npz")
        aligned = safe_set.density_data() is not None
        self.emit("safe_set_density_initial", status="ready" if aligned else "unaligned_safe_set",
                  original_attempts=len(np.unique(ids[:, 0])), original_states=len(ids),
                  matched_safe_set_rows=int(sum(np.sum(value[:, 0] != -1) for value in restored)))
        return aligned

    def _parts(self, safe_set):
        if self.separate_training:
            self._trim_training_history(safe_set)
        data = self._training_data(safe_set)
        if data is None:
            return None
        states, source_ids = data
        if self.safe_set_min_states and not self.separate_training:
            required = set(source_ids[:, 0])
            recent = [key for key in self.episodes if key >= 0]
            if recent:
                required.add(max(recent))  # Pending lap supplies the next extension.
            self.episodes = {key: value for key, value in self.episodes.items() if key in required}
        known = {tuple(key) for episode_ids in self.episodes.values() for key in episode_ids}
        if any(tuple(key) not in known for key in source_ids):
            return None
        if self.config.method == "sigma":
            # Assign on full measured attempts before restricting to retained
            # rows, preserving the sweep's roles across trimming and resume.
            roles = self._training_roles()
            assignments = np.asarray([roles[tuple(key)] for key in source_ids])
            values, ids = [], []
            for role in range(3):
                value, key = states[assignments == role], source_ids[assignments == role]
                if len(value) > self.config.max_states_per_partition:
                    chosen = np.sort(self.randomness.host.choice(
                        len(value), self.config.max_states_per_partition, replace=False))
                    value, key = value[chosen], key[chosen]
                values.append(value)
                ids.append(key)
            return values, ids
        # There is no NF holdout: all unique retained states train the model and
        # set its score bounds. Keep archive field names compatible; both views
        # reference the same data, and the independent test view is empty.
        return ([states, states, np.empty((0, states.shape[1]), dtype=states.dtype)],
                [source_ids, source_ids, np.empty((0, 2), dtype=source_ids.dtype)])

    def _training_data(self, safe_set):
        if not self.separate_training:
            return safe_set.density_data()
        # Keep the controller's existing first-lap warm-up lifecycle.
        history = self.training_history() if safe_set.frenet else None
        return None if history is None else (history["states"], history["ids"])

    def _data_signature(self, safe_set):
        data = self._training_data(safe_set)
        if data is None:
            return None
        states, ids = data
        digest = hashlib.sha256()
        for value in (ids.astype('<i8'), states.astype('<f8')):
            digest.update(value.tobytes())
        return digest.hexdigest()

    def fit(self, safe_set):
        parts = self._parts(safe_set)
        if parts is None:
            self.snapshot = self.local_calibration = None
            self.diagnostics = dict(status="held_unaligned_safe_set")
            self.emit("safe_set_density_status", **self.diagnostics)
            return False
        (training, calibration, test), ids = parts
        counts = dict(
            training_states=len(training),
            calibration_states=len(calibration),
            test_states=len(test),
        )
        fixed_steps = self.config.steps is not None
        # Empty partitions must reach the explicit empty-partition error below,
        # even when there is no usable retained-history record to inspect.
        retained_states = sum(counts.values())
        if len(training) and len(calibration):
            retained_states = len(self._training_data(safe_set)[0])
        target_training_states = (0 if fixed_steps else
                                  self.config.min_training_batches * self.config.batch_size)
        below_minimum = (
            len(training) < max(self.config.min_training_states, target_training_states)
            or len(calibration) < self.config.min_calibration_states
            or (fixed_steps and retained_states < self.safe_set_min_states)
        )
        minimums = dict(min_training_states=self.config.min_training_states,
                        min_calibration_states=self.config.min_calibration_states,
                        min_training_batches=self.config.min_training_batches,
                        target_training_states=target_training_states,
                        safe_set_min_states=self.safe_set_min_states,
                        retained_states=retained_states)
        if below_minimum:
            self.emit("safe_set_density_warning", reason="below_configured_minimum",
                      policy="continue_nf_density" if len(training) and len(calibration)
                      else "stop_empty_partition", **counts, **minimums)
        if not len(training) or not len(calibration):
            self.snapshot = self.local_calibration = None
            self.diagnostics = dict(status="empty_partition", **counts, **minimums)
            raise ValueError(
                "nf_density requires nonempty training and calibration partitions; "
                f"got {len(training)} training and {len(calibration)} calibration states. "
                "The selected safe-set method will not be replaced."
            )
        dtype = jnp.float64 if self.simulation.dtype == "float64" else jnp.float32
        # Each fresh model uses the current retained safe-set coordinates.
        # Publish these statistics with the completed model and its cutoffs;
        # all retained states contribute to normalization and calibration.
        offset = training[:, 1:5].mean(axis=0)
        scale = np.maximum(training[:, 1:5].std(axis=0), [0.05, 0.01, 0.5, 0.02])
        context = progress_context(jnp.asarray(training, dtype), self.track_length)
        targets = jnp.asarray((training[:, 1:5] - offset) / scale, dtype)
        # Each retained lap/progress bin has equal total training weight.
        bins = np.floor(
            np.remainder(training[:, 0], self.track_length)
            / self.track_length
            * self.config.progress_bins
        ).astype(int)
        _, inverse, counts_per_group = np.unique(
            np.column_stack((ids[0][:, 0], bins)), axis=0, return_inverse=True, return_counts=True
        )
        weights = len(training) / (len(counts_per_group) * counts_per_group[inverse])
        batch_size = self.config.batch_size
        total = ((len(training) + batch_size - 1) // batch_size) * batch_size
        batches_per_epoch = total // batch_size
        target_steps = (self.config.steps if fixed_steps
                        else self.config.epochs * batches_per_epoch)
        # A distinct initialization per published generation, reproducible
        # from the experiment seed even after checkpoint resume.
        initialization_key = jax.random.fold_in(self._initialization_key, self.generation + 1)
        candidate = initial_state(self.config, initialization_key, template=self.state)
        losses = []
        remaining_steps, fitting_samples = target_steps, 0
        while remaining_steps:
            pass_steps = min(remaining_steps, batches_per_epoch)
            order = self.randomness.host.permutation(len(training))
            order = np.pad(order, (0, total - len(order)), mode="wrap")
            mask = jnp.asarray(np.where(np.arange(total) < len(training), weights[order], 0), dtype)
            candidate, loss = train_epoch(
                candidate,
                context[order].reshape(-1, batch_size, 2)[:pass_steps],
                targets[order].reshape(-1, batch_size, 4)[:pass_steps],
                mask.reshape(-1, batch_size)[:pass_steps],
            )
            losses.append(float(loss))
            fitting_samples += min(len(training), pass_steps * batch_size)
            remaining_steps -= pass_steps
        if not np.isfinite(losses).all() or not all(
            np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(candidate.params)
        ):
            raise FloatingPointError("nonfinite safe-set density training")
        draft = DensitySnapshot(
            candidate,
            jnp.asarray(offset, dtype),
            jnp.asarray(scale, dtype),
            jnp.asarray(self.track_length, dtype),
            jnp.asarray(0.0, dtype),
            jnp.asarray(1.0, dtype),
            self.generation + 1,
            penalty_kind=density_penalty_kind(self.config),
            penalty_normalizer=jnp.asarray(np.log(2.), dtype),
        )
        calibration_scores = np.asarray(density_scores(draft, jnp.asarray(calibration, dtype)))
        cutoff, spread = calibrate_config(calibration_scores, self.config, dtype)
        draft = draft.replace(
            cutoff=jnp.asarray(cutoff, dtype), penalty_scale=jnp.asarray(spread, dtype),
            relaxation_cutoff=jnp.asarray(calibrate_relaxation(calibration_scores, self.config, dtype), dtype),
        )
        acceptance = float(relaxation_cutoff(draft))
        test_scores = (
            np.asarray(density_scores(draft, jnp.asarray(test, dtype)))
            if len(test)
            else np.empty(0)
        )
        per_bin = []
        cal_bins = np.floor(
            np.remainder(calibration[:, 0], self.track_length)
            / self.track_length
            * self.config.progress_bins
        ).astype(int)
        for index in range(self.config.progress_bins):
            scores = calibration_scores[cal_bins == index]
            per_bin.append(
                dict(
                    bin=index,
                    states=len(scores),
                    accepted_fraction=float(np.mean(scores <= acceptance)) if len(scores) else None,
                )
            )
        self.state, self.offset, self.scale = candidate, offset, scale
        self.generation += 1
        self.snapshot = inference_snapshot(draft)
        self.diagnostics = dict(
            status="ready",
            generation=self.generation,
            training_initialization="random_per_fit",
            feature_normalization="training_partition_per_fit",
            feature_offset=np.asarray(offset).tolist(),
            feature_scale=np.asarray(scale).tolist(),
            training_budget="fixed_steps" if fixed_steps else "epochs",
            epochs=len(losses),
            completed_epochs=target_steps // batches_per_epoch,
            effective_epochs=fitting_samples / len(training),
            fitting_samples=fitting_samples,
            configured_epochs=None if fixed_steps else self.config.epochs,
            configured_steps=self.config.steps,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            gradient_steps=int(candidate.step),
            learning_rate=self.config.learning_rate,
            cumulative_learning_rate=self.config.learning_rate * int(candidate.step),
            **counts,
            **minimums,
            below_minimum=below_minimum,
            cutoff=cutoff,
            penalty_scale=spread,
            nll_min=float(np.min(calibration_scores)) if self.config.method == "sigma" else cutoff,
            **density_method_metadata(self.config),
            zero_penalty_fraction=float(np.mean(calibration_scores <= cutoff)),
            penalty_transform=self.config.penalty_transform,
            calibration_coverage=float(np.mean(calibration_scores <= acceptance)),
            test_coverage=float(np.mean(test_scores <= acceptance)) if len(test_scores) else None,
            calibration_by_progress=per_bin,
            loss=losses[-1],
            losses=losses,
            split="random_states" if self.config.method == "sigma" else "all_safe_set_states",
            calibration_source="held_out_safe_set" if self.config.method == "sigma" else "training_safe_set",
            relaxation_cutoff=float(relaxation_cutoff(draft)),
            training_laps=len(self.safe_set_training) if self.separate_training else len(safe_set.frenet),
            data_source=self._data_source,
            calibration_mode=self.config.calibration,
            definition="Empirical successful-state retention, not safety probability or an invariance certificate",
        )
        self.diagnostics.update(upper_cutoff=cutoff + spread, penalty_kind=draft.penalty_kind)
        self.validation = dict(
            training_states=training,
            calibration_states=calibration,
            calibration_scores=calibration_scores,
            test_states=test,
            test_scores=test_scores,
            training_ids=ids[0],
            calibration_ids=ids[1],
            test_ids=ids[2],
            cutoff=np.asarray(cutoff),
        )
        self.data_signature = self._data_signature(safe_set)
        self.local_calibration = LocalDensityCalibration(
            self.snapshot, self.validation, None if self.separate_training else safe_set)
        self.save()
        self.emit("safe_set_density_training", **self.diagnostics)
        return True

    def warmup(self, counts):
        if self.snapshot is not None:
            snapshot = (LocalDensitySnapshot.create(self.snapshot, -np.inf, np.inf)
                        if self.config.calibration == "local" else self.snapshot)
            for count in sorted(set(counts)):
                density_penalty(
                    snapshot, jnp.zeros((count, 5), snapshot.offset.dtype)
                ).block_until_ready()
                if snapshot.relaxation_cutoff is not None:
                    density_violation(snapshot, jnp.zeros((count, 5), snapshot.offset.dtype)).block_until_ready()

    @property
    def _data_source(self):
        return ("independent_successful_post_states_v1" if self.separate_training else
                "retained_safe_set_post_states_v1")

    def _settings(self):
        density = {key: value for key, value in asdict(self.config).items()
                   if key not in ("calibration", "penalty_normalization",
                                  "relaxation_quantile")}
        if self.config.method == "sigma":
            density.pop("penalty_upperbound_quantile")
        else:
            for key in ("penalty_sigma", "sigma_cutoff_quantile", "max_states_per_partition"):
                density.pop(key)
        density["penalty_kind"] = density_penalty_kind(self.config)
        density["training_initialization"] = "random_per_fit_v1"
        density["feature_normalization"] = "training_partition_per_fit_v1"
        if self.config.steps is not None:
            # Epoch-only knobs cannot change the semantics of a fixed-step fit.
            density.pop("epochs")
            density.pop("min_training_batches")
            density["training_budget"] = "fixed_steps_v1"
        settings = dict(
            # Inference-window policy does not change the shared fitting data.
            # Existing aligned/global models can try local calibration directly.
            density=density,
            split=(dict(method="random_states_v1", calibration_fraction=.1, validation_fraction=0.)
                   if self.config.method == "sigma" else {"method": "all_safe_set_states_v1"}),
            seed=self.randomness.seed,
            track_length=self.track_length,
            dtype=self.simulation.dtype,
            keep=self.keep,
            data_source=self._data_source,
        )
        if self.config.steps is not None:
            settings["safe_set_min_states"] = self.safe_set_min_states
        return settings

    def save(self):
        if self.snapshot is None:
            return
        root = self.directory / "safe_set_density"
        root.mkdir(parents=True, exist_ok=True)
        payload = serialization.to_bytes({"state": self.state, "snapshot": self.snapshot})
        pending = root / ".model-pending"
        pending.write_bytes(payload)
        pending.replace(root / "model.msgpack")
        metadata = dict(
            schema=1,
            settings=self._settings(),
            diagnostics=self.diagnostics,
            generation=self.generation,
            attempt_ids=list(self.episodes),
            data_signature=self.data_signature,
            calibration={"calibration": self.config.calibration,
                         "penalty_normalization": self.config.penalty_normalization,
                         "selection": "sample_window" if self.config.calibration == "local" else "global",
                         **density_method_metadata(self.config),
                         "relaxation_cutoff": float(relaxation_cutoff(self.snapshot)),
                         "penalty_kind": self.snapshot.penalty_kind},
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        pending = root / ".metadata-pending"
        pending.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        pending.replace(root / "metadata.json")
        if self.validation is not None:
            np.savez_compressed(root / "validation.npz", **self.validation)
        history = self.training_history()
        if history is not None:
            pending = root / ".training-history-pending.npz"
            np.savez_compressed(pending, **history)
            pending.replace(root / "training_history.npz")

    def restore(self, checkpoint, safe_set):
        if self.separate_training:
            self._trim_training_history(safe_set)
        root = Path(checkpoint) / "safe_set_density"
        if not (root / "metadata.json").is_file():
            return False
        meta = json.loads((root / "metadata.json").read_text())
        if meta.get("schema") != 1:
            raise ValueError("unsupported safe-set density checkpoint")
        previous = meta["settings"]["density"]
        # Only an explicitly selected sigma run can reuse an older sigma fit.
        legacy_sigma = (self.config.method == "sigma" and "method" not in previous
                        and previous.get("penalty_rule") == "sigma")
        density = normalize_density_settings(previous, method="sigma" if legacy_sigma else None)
        density.setdefault("method", "sigma" if legacy_sigma else "quantile")
        if density["method"] == "sigma":
            density.pop("penalty_upperbound_quantile", None)
        else:
            for key in ("penalty_sigma", "sigma_cutoff_quantile"):
                density.pop(key, None)
        settings = {**meta["settings"], "density": density}
        settings["density"].setdefault("penalty_transform", "linear")
        settings["density"].setdefault("min_training_batches", 0)
        settings["density"].setdefault("steps", None)
        if settings["density"]["steps"] is not None:
            settings["density"].pop("min_training_batches", None)
        if (settings != self._settings()
            or meta["attempt_ids"] != list(self.episodes)
            or meta.get("data_signature") != self._data_signature(safe_set)):
            self.generation = int(meta["generation"])
            self.emit("safe_set_density_status", status="rebuilding_changed_settings_or_data")
            return False
        payload = (root / "model.msgpack").read_bytes()
        if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
            raise ValueError("safe-set density checkpoint checksum differs")
        dtype = jnp.float64 if self.simulation.dtype == "float64" else jnp.float32
        template = inference_snapshot(
            DensitySnapshot(
                self.state,
                jnp.zeros(4, dtype),
                jnp.ones(4, dtype),
                jnp.asarray(self.track_length, dtype),
                jnp.asarray(0.0, dtype),
                jnp.asarray(1.0, dtype),
                0,
                penalty_kind=density_penalty_kind(self.config),
                penalty_normalizer=jnp.asarray(np.log(2.), dtype),
            )
        )
        saved = serialization.msgpack_restore(payload)
        # Linear checkpoints predate the log normalization leaf.
        saved["snapshot"].setdefault("penalty_normalizer", np.asarray(np.log(2.), dtype))
        # Recompute both bounds: old checkpoints may contain a retired penalty
        # cutoff. The fitted NF, feature normalization and optimizer are reused.
        saved["snapshot"]["relaxation_cutoff"] = None
        restored = serialization.from_state_dict({"state": self.state, "snapshot": template}, saved)
        restored = jax.tree.map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, restored
        )
        self.state, self.snapshot = restored["state"], restored["snapshot"]
        self.generation = int(self.snapshot.generation)
        self.offset, self.scale = np.asarray(self.snapshot.offset), np.asarray(self.snapshot.scale)
        self.diagnostics = {**normalize_density_settings(meta["diagnostics"], method=self.config.method),
                            "calibration_mode": self.config.calibration}
        self.data_signature = meta["data_signature"]
        self.validation = self.local_calibration = None
        if (root / "validation.npz").is_file():
            with np.load(root / "validation.npz", allow_pickle=False) as data:
                self.validation = {name: data[name].copy() for name in data.files}
            scores = self.validation["calibration_scores"]
            minimum, width = calibrate_config(scores, self.config, dtype)
            bound = calibrate_relaxation(scores, self.config, dtype)
            self.snapshot = self.snapshot.replace(
                cutoff=jnp.asarray(minimum, dtype), penalty_scale=jnp.asarray(width, dtype),
                penalty_normalizer=jnp.asarray(np.log(2.), dtype),
                relaxation_cutoff=jnp.asarray(bound, dtype))
            acceptance = float(relaxation_cutoff(self.snapshot))
            self.validation["cutoff"] = np.asarray(minimum)
            self.diagnostics.update(cutoff=minimum,
                                    nll_min=float(np.min(scores)) if self.config.method == "sigma" else minimum,
                                    **density_method_metadata(self.config),
                                    penalty_scale=width, upper_cutoff=minimum + width,
                                    zero_penalty_fraction=float(np.mean(scores <= minimum)),
                                    calibration_coverage=float(np.mean(scores <= acceptance)))
            progress = np.remainder(self.validation["calibration_states"][:, 0], self.track_length)
            bins = np.floor(progress / self.track_length * self.config.progress_bins).astype(int)
            self.diagnostics["calibration_by_progress"] = [
                dict(bin=index, states=int(np.sum(bins == index)),
                     accepted_fraction=(float(np.mean(scores[bins == index] <= acceptance))
                                        if np.any(bins == index) else None))
                for index in range(self.config.progress_bins)]
            self.local_calibration = LocalDensityCalibration(
                self.snapshot, self.validation, None if self.separate_training else safe_set)
        else:
            raise ValueError("NF penalty calibration requires saved safe-set calibration scores")
        self.diagnostics.update(**density_method_metadata(self.config),
                                relaxation_cutoff=float(relaxation_cutoff(self.snapshot)))
        self.save()
        self.emit("safe_set_density_status", status="restored", generation=self.generation)
        return True
