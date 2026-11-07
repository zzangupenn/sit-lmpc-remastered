"""Read-only NF score diagnostics for the option-6 safe-set view."""

import jax.numpy as jnp
import numpy as np

from sit_lmpc.training.safe_set_density import calibration_bound, density_scores, density_method_metadata


def calibration_display_aliases(safe_set, states, validation, dtype):
    """Show calibrated identities at every displayed extension position.

    This is display-only: all calibration statistics still use the original
    unique reference arrays. Each alias keeps its identity's cached NLL score.
    """
    calibration = np.asarray(validation["calibration_states"])
    scores = np.asarray(validation["calibration_scores"])
    progress, displayed_scores = calibration[:, 0].tolist(), scores.tolist()
    ids = validation.get("calibration_ids")
    if (ids is not None and safe_set.state_ids and len(safe_set.state_ids) == len(safe_set.frenet)
            and all(len(geometry_ids) == len(record) for geometry_ids, record in
                    zip(safe_set.state_ids, safe_set.frenet, strict=True))):
        lookup = {tuple(identity): index for index, identity in enumerate(ids)}
        # Compare progress as actually presented to NF inference; insignificant
        # host/device precision differences are not distinct extension copies.
        seen = {(*identity, float(np.asarray(s, dtype=dtype)))
                for identity, s in zip(ids, progress, strict=True)}
        for identity, state in zip(np.concatenate(safe_set.state_ids), states, strict=True):
            row = lookup.get(tuple(identity))
            key = (*identity, float(np.asarray(state[0], dtype=dtype)))
            if row is not None and key not in seen:
                progress.append(float(state[0]))
                displayed_scores.append(float(scores[row]))
                seen.add(key)
    return np.asarray(progress), np.asarray(displayed_scores)


class DensityCalibrationDiagnostics:
    """Cache retained-state scores per immutable density generation/geometry."""

    def __init__(self):
        self._key = self._snapshot = self._records = self._pool = None

    def evaluate(self, result, safe_set, snapshot, validation, config, local_cache=None, *, training=None):
        generation = None if snapshot is None else int(snapshot.generation)
        loss = (training.get("loss") if training is not None
                and generation is not None and training.get("generation") == generation else None)
        loss = float(loss) if loss is not None and np.isfinite(loss) else None
        model_info = dict(training_loss=loss, training_generation=generation)
        settings = density_method_metadata(config.safe_set_density)
        method = result.metrics.get("safe_set_density_method", settings["method"])
        relaxation_quantile = result.metrics.get("safe_set_density_relaxation_quantile",
                                                 settings["relaxation_quantile"])
        upper_quantile = result.metrics.get("safe_set_density_penalty_upperbound_quantile",
                                            settings["penalty_upperbound_quantile"])
        method_info = dict(method=method, relaxation_quantile=relaxation_quantile,
                           penalty_upperbound_quantile=upper_quantile,
                           **(dict(penalty_sigma=result.metrics.get("safe_set_density_penalty_sigma",
                                                                   config.safe_set_density.penalty_sigma))
                              if method == "sigma" else {}))
        unavailable = dict(penalty_transform=config.safe_set_density.penalty_transform, **method_info, **model_info)
        if snapshot is None or validation is None:
            message = "NF calibration is not ready yet."
            if snapshot is None and training is not None:
                status = training.get("status")
                if status == "held_insufficient_data":
                    density = config.safe_set_density
                    message = (
                        "NF unavailable: this recording has no fitted model.\n"
                        f"Training states: {training['training_states']} "
                        f"(warning minimum {density.min_training_states})\n"
                        f"Calibration states: {training['calibration_states']} "
                        f"(warning minimum {density.min_calibration_states})"
                    )
                elif status == "held_unaligned_safe_set":
                    message = "NF unavailable: retained safe-set states are not aligned."
            return dict(status="unavailable", message=message, **unavailable)
        if result.debug is None:
            return dict(status="unavailable", message="Sampled terminal states appear when learned control starts.", **unavailable)
        key = id(snapshot), safe_set.signature, id(validation)
        if key != self._key:
            states = np.asarray(safe_set.arrays()[0][:, :5])
            scores = np.asarray(density_scores(snapshot, jnp.asarray(states, snapshot.offset.dtype)))
            self._pool = dict(safe_set_states=states.copy(), safe_set_scores=scores.copy(),
                              calibration_states=np.asarray(validation["calibration_states"]).copy(),
                              calibration_scores=np.asarray(validation["calibration_scores"]).copy())
            progress, displayed_scores = calibration_display_aliases(
                safe_set, states, validation, snapshot.offset.dtype)
            self._pool.update(calibration_display_s=progress,
                              calibration_display_scores=displayed_scores)
            self._key, self._snapshot = key, snapshot
            self._records = tuple(safe_set.frenet), tuple(safe_set.cartesian), validation
        paths = np.asarray(result.debug["states"])
        valid = np.asarray(result.debug.get("valid_samples", np.isfinite(paths).all(axis=(1, 2)))).copy()
        valid &= np.isfinite(paths).all(axis=(1, 2))
        if not np.any(valid):
            return dict(status="unavailable", message="No valid sampled terminal states in this update.", **unavailable)
        terminal = paths[valid, -1, :5].copy()
        terminal_scores = np.asarray(density_scores(snapshot, jnp.asarray(terminal, snapshot.offset.dtype)))
        pool = self._pool
        length = float(snapshot.track_length)
        cal_states, cal_scores = pool["calibration_states"], pool["calibration_scores"]
        bins = config.safe_set_density.progress_bins
        phase = np.remainder(cal_states[:, 0], length)
        bin_numbers = np.minimum((phase / length * bins).astype(int), bins - 1)
        cuts = np.full(bins, np.nan)
        for index in range(bins):
            values = cal_scores[bin_numbers == index]
            if len(values):
                cuts[index] = calibration_bound(values, relaxation_quantile, snapshot.offset.dtype)
        mode = result.metrics.get("safe_set_density_calibration", config.safe_set_density.calibration)
        active = result.metrics.get("safe_set_query_active") == "nf_density"
        # Reuse the actual pre-lambda costs, including local support and numeric
        # saturation, rather than infer penalties from the plotted NLL lines.
        terminal_penalties = np.asarray(result.debug["hull"])[valid, -1].copy() if active else None
        cutoff = result.metrics.get("safe_set_density_cutoff") if active else None
        if active and cutoff is None and mode == "global":
            cutoff = float(snapshot.cutoff)
        relaxation = result.metrics.get("safe_set_density_relaxation_cutoff") if active else None
        if active and relaxation is None:
            relaxation = result.debug.get("density_relaxation_cutoff")
        relaxation = None if relaxation is None else float(relaxation)
        origin = result.metrics.get("safe_set_density_penalty_origin", "legacy_cutoff")
        upper = result.metrics.get("safe_set_density_upper_cutoff") if active else None
        if active and upper is None and mode == "global":
            upper = float(snapshot.cutoff + snapshot.penalty_scale)
        transform = result.metrics.get("safe_set_density_penalty_transform",
                                       "log1p" if snapshot.penalty_kind == "log1p" else "linear")
        normalization = result.metrics.get("safe_set_density_penalty_normalization", "calibrated")
        normalizer = result.debug.get("density_penalty_normalizer",
                                     result.metrics.get("safe_set_density_log_normalizer"))
        if normalizer is None and normalization == "calibrated":
            normalizer = snapshot.penalty_normalizer
        normalizer = (float(normalizer) if normalizer is not None
                      and np.isfinite(normalizer) and normalizer > 0 else None)
        if transform == "log1p" and result.metrics.get("safe_set_density_sample_max_active") == 0:
            upper = None
        window = None
        selected_s, selected_scores = cal_states[:, 0], cal_scores
        if mode == "local":
            selected_s, selected_scores = np.empty(0), np.empty(0)
            lo, hi = (result.metrics.get(f"safe_set_density_window_{side}") for side in ("min", "max"))
            if lo is not None and hi is not None:
                window = float(lo), float(hi)
                if (local_cache is not None and local_cache.snapshot is snapshot
                        and (local_cache.signature is None or local_cache.signature == safe_set.signature)):
                    rows = np.flatnonzero((local_cache.progress > lo) & (local_cache.progress < hi))
                    _, first = np.unique(local_cache.identities[rows], return_index=True)
                    selected_s = local_cache.progress[rows[first]].copy()
                    selected_scores = local_cache.scores[rows[first]].copy()
        setting = np.empty(0, dtype=int)
        if cutoff is not None and len(selected_scores):
            if origin == "safe_set_min":
                setting = np.flatnonzero(selected_scores == np.min(selected_scores))
            else:
                setting = np.flatnonzero(selected_scores >= cutoff)
        return dict(status="ready", **model_info, **pool, terminal_states=terminal, terminal_scores=terminal_scores,
                    excluded_samples=int(np.sum(~valid)),
                    terminal_penalties=terminal_penalties,
                    track_length=length, coverage=relaxation_quantile,
                    section_cutoffs=cuts, global_cutoff=float(snapshot.cutoff),
                    active_method=result.metrics.get("safe_set_query_active", "unavailable"),
                    calibration_mode=mode, active_cutoff=cutoff, active_window=window,
                    active_relaxation_cutoff=relaxation,
                    **method_info,
                    penalty_origin=origin, active_upper_cutoff=upper,
                    penalty_transform=transform,
                    compression_scale=result.metrics.get("safe_set_density_penalty_scale", float(snapshot.penalty_scale)),
                    penalty_normalizer=normalizer,
                    calibrated_upper_cutoff=result.metrics.get("safe_set_density_calibrated_upper_cutoff"),
                    penalty_normalization=normalization,
                    selected_calibration_s=np.asarray(selected_s),
                    selected_calibration_scores=np.asarray(selected_scores), setting_indices=setting,
                    calibration_status=result.metrics.get("safe_set_density_calibration_status", "ready"))
