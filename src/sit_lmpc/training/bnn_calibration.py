"""Blocked, experiment-local calibration of a fixed Gaussian function model.

Likelihood scales are working model-error scales, not injected process noise.
Only calibration data selects a candidate; test outcomes never select updates.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from dyna_dist.inference.posterior import FunctionSamples, components
from sit_lmpc.data.dynamics import RolloutBatch, TransitionBatch
from sit_lmpc.dynamics.gaussian_bnn import sample_coefficients


def split_calibrated(batch, config, seed=0):
    """Shared random sliding-window split, or the explicit legacy contiguous split."""
    from sit_lmpc.data.partitions import partition_indices

    return tuple(batch.take(rows) for rows in partition_indices(batch, config, seed))


def calibration_windows(batch, horizon, limit, *, sliding=False, seed=0):
    """Complete windows; random sliding starts or legacy non-overlapping windows."""
    starts = []
    for episode in np.unique(batch.ids[:, 0]):
        rows = np.flatnonzero(batch.ids[:, 0] == episode)
        i = 0
        while i + horizon <= len(rows):
            section = rows[i:i + horizon]
            if (np.all(np.diff(batch.ids[section, 1]) == 1)
                    and np.array_equal(batch.states[section[1:]], batch.next_states[section[:-1]])):
                starts.append(section)
                i += 1 if sliding else horizon
            else:
                i += 1
    if len(starts) > limit:
        indices = (np.sort(np.random.default_rng(seed).choice(len(starts), limit, replace=False))
                   if sliding else np.linspace(0, len(starts)-1, limit).astype(int))
        starts = [starts[i] for i in indices]
    rows = np.asarray(starts, dtype=int).reshape(-1, horizon)
    return RolloutBatch(
        batch.states[rows[:, 0]], batch.controls[rows], batch.next_states[rows],
        batch.intervals[rows], batch.substeps[rows], np.ones(rows.shape, dtype=bool),
    )


@jax.jit
def _score_paths(model, coefficients, states, controls, targets, mask):
    paths = FunctionSamples(model, coefficients).rollout(states, controls)[..., :2]
    center = jnp.mean(paths, axis=1)
    error = jnp.linalg.norm(center - targets[..., :2], axis=-1)
    radius = jnp.quantile(
        jnp.max(jnp.linalg.norm(paths - center[:, None], axis=-1), axis=-1),
        .95, axis=1, method="higher",
    )
    flattened = paths.reshape(paths.shape[0], paths.shape[1], -1) / jnp.sqrt(paths.shape[2])
    truth = targets[..., :2].reshape(paths.shape[0], -1) / jnp.sqrt(paths.shape[2])
    first = jnp.mean(jnp.linalg.norm(flattened - truth[:, None], axis=-1), axis=1)
    second = jnp.mean(jnp.linalg.norm(flattened[:, :, None] - flattened[:, None], axis=-1), axis=(1, 2))
    count = jnp.maximum(mask.sum(), 1)
    average = lambda value: jnp.sum(jnp.where(mask, value, 0)) / count
    return jnp.array([
        jnp.all(jnp.isfinite(paths)), jnp.sqrt(average(error[:, -1] ** 2)),
        average(jnp.max(error, axis=-1) <= radius), average(radius), average(first - .5 * second),
    ])


def score_windows(model, mean, factor, windows, config, key):
    if not len(windows):
        return {"status": "insufficient_windows", "windows": 0}
    n, size, dtype = len(windows), config.validation_max_windows, model.normalizer.target_scale.dtype
    if n > size:
        raise ValueError("calibration windows exceed the configured fixed batch")
    indices = np.pad(np.arange(n), (0, size - n), mode="edge")
    coefficients = sample_coefficients(jnp.asarray(mean, dtype), jnp.asarray(factor, dtype),
                                       key, config.validation_particles)
    values = np.asarray(jax.device_get(_score_paths(
        model, coefficients, jnp.asarray(windows.states[indices], dtype),
        jnp.asarray(windows.controls[indices], dtype), jnp.asarray(windows.targets[indices], dtype),
        jnp.arange(size) < n,
    )), dtype=float)
    if not np.isfinite(values).all() or not values[0]:
        return {"status": "nonfinite_rollouts", "windows": n}
    return {"status": "ready", "windows": n,
            **dict(zip(("endpoint_xy_rmse", "path_coverage_95", "path_radius_95", "energy_score"),
                       values[1:].tolist(), strict=True))}


@jax.jit
def _residuals(model, states, controls, targets, mean):
    base, basis = components(model, states, controls)
    return model.targets(states, controls, targets) - base - basis @ mean


def estimate_variance(model, evidence, calibration, config):
    """Out-of-fit calibration residual MSE, floored at pretrained variance."""
    mean = evidence.moments(config.likelihood_power)[0]
    squared = np.zeros_like(evidence.variance)
    size, dtype = config.update_batch_size, model.normalizer.target_scale.dtype
    for start in range(0, len(calibration), size):
        count = min(size, len(calibration) - start)
        arrays = []
        for data, width in ((calibration.states, 7), (calibration.controls, 2),
                            (calibration.next_states, 7)):
            padded = np.zeros((size, width), dtype=np.dtype(dtype))
            padded[:count] = data[start:start + count]
            arrays.append(jnp.asarray(padded))
        residual = np.asarray(jax.device_get(_residuals(model, *arrays, jnp.asarray(mean, dtype))), float)
        squared += np.sum(residual[:count] ** 2, axis=0)
    variance = np.maximum(evidence.variance, squared / max(len(calibration), 1))
    if not np.isfinite(variance).all():
        raise FloatingPointError("nonfinite calibrated working likelihood scales")
    return variance


@dataclass(frozen=True)
class CalibrationDecision:
    mean: np.ndarray | None
    factor: np.ndarray | None
    variance: np.ndarray | None
    power: float | None
    diagnostics: dict


def select_update(model, evidence, calibration, previous, config, key, *, method=None):
    """Compare candidates on calibration only; returns no posterior on a hold."""
    method = config.posterior_method if method is None else method
    sliding = config.calibration_split == "random_windows"
    window_seed = np.asarray(jax.random.key_data(jax.random.fold_in(key, 411))).tolist() if sliding else 0
    windows = calibration_windows(calibration, config.calibration_horizon, config.validation_max_windows,
                                  sliding=sliding, seed=window_seed)
    diagnostics = {"method": method, "horizon": config.calibration_horizon, "windows": len(windows),
                   "split": config.calibration_split,
                   "window_sampling": "random_sliding" if sliding else "nonoverlapping",
                   "target_coverage": config.calibration_target_coverage, "candidates": [],
                   "definition": "95% function-only XY path tube; empirical calibration, not a safety guarantee"}
    if len(windows) < config.calibration_min_windows:
        return CalibrationDecision(None, None, None, None,
                                   {**diagnostics, "status": "held_insufficient_calibration"})
    reference = score_windows(model, previous.mean, previous.factor, windows, config, key)
    diagnostics["previous"] = reference
    variance = (estimate_variance(model, evidence, calibration, config)
                if method in ("noise", "noise_tempered") else evidence.variance)
    powers = ([min(config.likelihood_power, x) for x in config.calibration_powers]
              if method in ("tempered", "noise_tempered") else [config.likelihood_power])
    best = None
    for power in sorted(set(powers), reverse=True):
        mean, factor = evidence.moments(power, variance)
        scores = score_windows(model, mean, factor, windows, config, key)
        finite = scores["status"] == "ready"
        coverage_ok = finite and scores["path_coverage_95"] >= config.calibration_target_coverage
        score_ok = (finite and (reference["status"] != "ready" or
                    scores["energy_score"] <= reference["energy_score"] * (1 + config.calibration_score_slack)))
        diagnostics["candidates"].append({"power": power, **scores,
                                           "eligible": bool(coverage_ok and score_ok)})
        if coverage_ok and score_ok and (best is None or scores["energy_score"] < best[-1]["energy_score"]):
            best = (mean, factor, variance, power, scores)
    diagnostics["working_variance"] = variance.tolist()
    if best is None:
        return CalibrationDecision(None, None, None, None,
                                   {**diagnostics, "status": "held_no_calibrated_candidate"})
    diagnostics.update(status="ready", selected=best[-1], selected_power=best[3])
    return CalibrationDecision(*best[:4], diagnostics)
