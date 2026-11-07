"""Standalone MPPI diagnostics using fixed functions and exact Gym ST references.

The local tracking task deliberately has no BRO map, learned value, risk-to-go,
or safe-set hull. Its identical candidate controls isolate dynamics quality.
"""

from functools import lru_cache
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from dyna_dist.io.common import write_json


@lru_cache(maxsize=1)
def _gym_batch_kernel():
    from numba import njit, prange, get_num_threads, set_num_threads
    from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st as rhs

    set_num_threads(min(8, get_num_threads()))

    @njit(parallel=True)
    def batch(initial, controls, parameters, dt, substep):
        candidates, horizon = controls.shape[:2]
        count = len(parameters)
        paths = np.empty((candidates, count, horizon + 1, 7), dtype=np.float64)
        for index in prange(candidates * count):
            c, p = index // count, index % count
            state = initial.copy()
            paths[c, p, 0] = state
            for step in range(horizon):
                action = controls[c, step]
                for _ in range(int(dt / substep)):
                    k1 = rhs(state, action, parameters[p])
                    k2 = rhs(state + substep * (k1 / 2), action, parameters[p])
                    k3 = rhs(state + substep * (k2 / 2), action, parameters[p])
                    k4 = rhs(state + substep * k3, action, parameters[p])
                    state = state + substep * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                state[4] = (state[4] + np.pi) % (2 * np.pi) - np.pi
                paths[c, p, step + 1] = state
        return paths

    return batch


def physical_candidates(simulators, initial, controls, dataset):
    return _gym_batch_kernel()(
        np.asarray(initial, dtype=np.float64), np.asarray(controls, dtype=np.float64),
        np.stack([s.params for s in simulators]), dataset.timestep,
        dataset.manifest["config"]["integrator_timestep"],
    )


def sample_controls(nominal, key, count, config):
    """Same truncated perturbation rule and physical units as BRO LMPPI."""
    scale = jnp.asarray(config.mppi_action_scale)
    center = jnp.clip(jnp.asarray(nominal) / scale, -1, 1)
    std = jnp.asarray(config.mppi_sample_std)
    delta = jax.random.truncated_normal(
        key, (-1 - center) / std, (1 - center) / std,
        shape=(count, len(center), 2),
    ) * std
    actions = jnp.clip(center + delta, -1, 1) * scale
    return np.asarray(actions.at[0].set(center * scale))


def path_scores(paths, reference, controls, widths):
    """Expected tracking cost and probability of ANY corridor exit in x_1..x_H."""
    paths = np.asarray(paths)
    xy = paths[:, :, 1:, :2]
    target = reference[1:]
    heading = target[:, 4]
    normals = np.column_stack((-np.sin(heading), np.cos(heading)))
    difference = xy - target[None, None, :, :2]
    lateral = np.sum(difference * normals, axis=-1)
    heading_error = (paths[:, :, 1:, 4] - heading + np.pi) % (2 * np.pi) - np.pi
    tracking = np.sum(difference**2, axis=-1)
    tracking += 0.2 * (paths[:, :, 1:, 3] - target[:, 3]) ** 2
    tracking += 0.1 * heading_error**2
    finite = np.isfinite(paths[:, :, 1:]).all(axis=(2, 3))
    costs = np.where(finite, np.mean(tracking, axis=-1), 1e6)
    rate = 0.01 * np.mean(np.sum(np.diff(controls, axis=1)**2, axis=-1), axis=-1)
    if controls.shape[1] == 1:
        rate = np.zeros(len(controls))
    probabilities = np.stack([
        ((np.abs(lateral) > width).any(axis=-1) | ~finite).mean(axis=1)
        for width in widths
    ], axis=-1)
    costs = np.minimum(np.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=1e6), 1e6)
    expected = costs.mean(axis=1) + rate
    # A fixed cost definition shared by learned and physical evaluators.
    objective = expected + 10 * probabilities[:, len(widths) // 2]
    return {"cost": expected, "objective": objective, "probabilities": probabilities,
            "nonfinite_fraction": float(1 - finite.mean())}


def optimized_controls(controls, objective, config):
    score = np.asarray(objective)
    normalized = -(score - score.min()) / (np.ptp(score) + 0.001)
    weights = np.exp(normalized / config.mppi_temperature)
    weights /= weights.sum()
    return np.einsum("c,cha->ha", weights, controls)


def compare_decisions(prediction, reference, threshold):
    predicted = prediction["objective"]
    truth = reference["objective"]
    count = max(1, len(truth) // 10)
    top_a, top_b = np.argsort(predicted)[:count], np.argsort(truth)[:count]
    rank = spearmanr(predicted, truth).statistic
    selected = int(np.argmin(predicted))
    safe = prediction["probabilities"] <= threshold
    unsafe = reference["probabilities"] > threshold
    false_safe = safe & unsafe
    return {
        "spearman_rank": float(rank) if np.isfinite(rank) else None,
        "top_10_percent_overlap": float(len(np.intersect1d(top_a, top_b)) / count),
        "selected_candidate": selected,
        "selected_reference_objective": float(truth[selected]),
        "selected_candidate_cost_gap": float(truth[selected] - truth.min()),
        "probability_mae_by_width": np.mean(np.abs(
            prediction["probabilities"] - reference["probabilities"]
        ), axis=0).tolist(),
        "false_safe_fraction_of_all_by_width": false_safe.mean(axis=0).tolist(),
        "false_safe_fraction_of_predicted_safe_by_width": [
            float(false_safe[:, i].sum() / safe[:, i].sum()) if safe[:, i].any() else None
            for i in range(safe.shape[1])
        ],
        "predicted_safe_count_by_width": safe.sum(axis=0).tolist(),
        "nonfinite_fraction": prediction["nonfinite_fraction"],
    }


def _learned_paths(functions, initial, controls):
    learned = np.asarray(functions.rollout(initial, controls))
    start = np.broadcast_to(initial, (*learned.shape[:2], 1, 7))
    return np.concatenate((start, learned), axis=2)


def _benchmark(models, initial, controls, config, emit):
    rows = []
    for latent, model in models.items():
        for count in config.benchmark_particles:
            functions = model.sample_functions(jax.random.key(config.seed), count)
            started = time.perf_counter()
            functions.rollout(initial, controls).block_until_ready()
            compilation = time.perf_counter() - started
            times = []
            for _ in range(3):
                started = time.perf_counter()
                functions.rollout(initial, controls).block_until_ready()
                times.append(time.perf_counter() - started)
            rows.append({
                "latent": latent, "particles": count, "candidates": len(controls),
                "horizon": config.horizon, "first_call_seconds": compilation,
                "steady_state_ms": float(np.median(times) * 1000),
                "rollouts_per_second": float(len(controls) * count / np.median(times)),
            })
            emit(f"[vehicle-mppi] latent {latent}, {count} functions: "
                 f"{rows[-1]['steady_state_ms']:.1f} ms prediction")
    return {"devices": [str(device) for device in jax.devices()], "rows": rows,
            "scope": "fixed-function rollout kernel only; excludes costs, hulls, transfers, and I/O"}


def _replanning(models, dataset, config, root, sampled, simulators, emit):
    from .vehicle_bnn import fresh_probe, _simulators

    rows = []
    # One seen and one unseen real dynamics instance; both reuse fresh controls.
    for split in ("train", "test"):
        member = dataset.members(split)[0]
        real = _simulators(dataset, np.array([[
            member[key] for key in ("friction", "front_cornering_scale",
                                   "rear_cornering_scale", "mass_ratio")
        ]]))[0]
        speed = config.rollout_speeds[len(config.rollout_speeds) // 2]
        tape, target = fresh_probe(dataset, config, speed, 99, extra_steps=config.replan_steps)
        start = max(config.rollout_starts)
        reference = target[start:start + config.replan_steps + config.horizon + 1]
        nominal_initial = tape[start:start + config.horizon]
        paths = {}
        for name, functions in [("physical", None)] + [
            (f"latent_{latent}", sampled[latent]) for latent in models
        ]:
            state = reference[0].copy()
            nominal = nominal_initial.copy()
            actual = [state.copy()]
            actions = []
            for step in range(config.replan_steps):
                controls = sample_controls(
                    nominal, jax.random.fold_in(jax.random.key(config.seed + 700), step),
                    config.replan_candidates, config,
                )
                prediction = (physical_candidates(simulators, state, controls, dataset)
                              if functions is None else _learned_paths(functions, state, controls))
                score = path_scores(prediction, reference[step:step + config.horizon + 1],
                                    controls, config.mppi_half_widths)
                optimal = optimized_controls(controls, score["objective"], config)
                state = real.step(state, optimal[0])
                actual.append(state.copy())
                actions.append(optimal[0])
                nominal = np.concatenate((optimal[1:], optimal[-1:]))
            actual = np.asarray(actual)
            paths[name] = actual
            target_xy = reference[:len(actual), :2]
            error = actual[:, :2] - target_xy
            heading = reference[:len(actual), 4]
            lateral = -np.sin(heading) * error[:, 0] + np.cos(heading) * error[:, 1]
            rows.append({
                "split": split, "instance_id": member["instance_id"], "model": name,
                "steps": config.replan_steps,
                "position_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
                "max_cross_track_m": float(np.max(np.abs(lateral))),
                "terminal_position_error_m": float(np.linalg.norm(error[-1])),
            })
            np.savez_compressed(root / "mppi" / f"replan-{split}-{name}.npz",
                                states=actual, actions=actions,
                                reference=reference[:len(actual)])
            emit(f"[vehicle-mppi] {split} {name}: {config.replan_steps} replanning steps complete")
    return rows


def evaluate_mppi(models, dataset, config, root, windows, emit):
    from .vehicle_bnn import _simulators

    directory = root / "mppi"
    directory.mkdir(exist_ok=True)
    with np.load(root / "sampled_functions.npz", allow_pickle=False) as saved:
        values = saved["real_parameter_values"][:config.mppi_particles]
    simulators = _simulators(dataset, values)
    sampled = {latent: model.sample_functions(jax.random.key(config.seed), config.mppi_particles)
               for latent, model in models.items()}
    np.savez_compressed(
        directory / "function_samples.npz", real_parameter_values=values,
        **{f"latent_{latent}_coefficients": np.asarray(functions.coefficients)
           for latent, functions in sampled.items()},
    )
    rows = []
    for window_index, window in enumerate(windows):
        if window["signal"] != config.rollout_signals[0]:
            continue
        probe_id = window["probe_id"]
        with np.load(root / "rollouts" / f"{probe_id}.npz", allow_pickle=False) as shard:
            nominal = shard["controls"]
            reference = shard["real_model_paths"].mean(axis=0)
        initial = np.asarray(window["initial_state"])
        controls = sample_controls(nominal, jax.random.fold_in(jax.random.key(config.seed),
                                                             window_index),
                                   config.mppi_candidates, config)
        real_paths = physical_candidates(simulators, initial, controls, dataset)
        real_scores = path_scores(real_paths, reference, controls, config.mppi_half_widths)
        arrays = {"controls": controls, "reference": reference,
                  "reference_objective": real_scores["objective"],
                  "reference_probabilities": real_scores["probabilities"]}
        real_optimal = optimized_controls(controls, real_scores["objective"], config)
        real_proposal = path_scores(physical_candidates(
            simulators, initial, real_optimal[None], dataset
        ), reference, real_optimal[None], config.mppi_half_widths)
        for latent, functions in sampled.items():
            learned = _learned_paths(functions, initial, controls)
            score = path_scores(learned, reference, controls, config.mppi_half_widths)
            comparison = compare_decisions(score, real_scores, config.mppi_probability_threshold)
            optimal = optimized_controls(controls, score["objective"], config)
            physical_optimal = physical_candidates(simulators, initial, optimal[None], dataset)
            proposal = path_scores(physical_optimal, reference, optimal[None], config.mppi_half_widths)
            comparison.update({
                "latent": latent, "probe_id": probe_id, "speed_mps": window["speed_mps"],
                "start_step": window["start_step"],
                "weighted_proposal_reference_objective": float(proposal["objective"][0]),
                "reference_weighted_proposal_objective": float(real_proposal["objective"][0]),
                "weighted_proposal_cost_gap": float(proposal["objective"][0]
                                                   - real_proposal["objective"][0]),
                "weighted_proposal_reference_probabilities": proposal["probabilities"][0].tolist(),
                "first_action_difference": np.abs(optimal[0] - real_optimal[0]).tolist(),
            })
            rows.append(comparison)
            arrays.update({f"latent_{latent}_objective": score["objective"],
                           f"latent_{latent}_probabilities": score["probabilities"],
                           f"latent_{latent}_proposal": optimal,
                           f"latent_{latent}_proposal_true_paths": physical_optimal[0]})
        np.savez_compressed(directory / f"{probe_id}.npz", **arrays)
        emit(f"[vehicle-mppi] {probe_id}: {len(controls)} paired candidate decisions complete")
    benchmark = _benchmark(models, initial, controls, config, emit)
    replanning = _replanning(models, dataset, config, root, sampled, simulators, emit)
    result = {"rows": rows, "benchmark": benchmark, "replanning": replanning,
              "reference_particles": len(simulators),
              "task": "local tracking with squared XY error + 0.2 speed error squared + "
                      "0.1 wrapped heading error squared + 0.01 action-rate penalty; "
                      "add 10 times any-step corridor-exit probability at the middle half-width",
              "probability_resolution": 1 / len(simulators),
              "reference_population": "uniform independent parameter ranges via Latin-hypercube draws; "
                                      "an interpolation target, not an identified posterior"}
    write_json(directory / "metrics.json", result)
    return result
