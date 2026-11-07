"""Prior-only vehicle function-distribution evaluation against physical ST samples."""

from dataclasses import fields, replace
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.config import (
    VehicleBNNEvaluationConfig, VehicleCollectionConfig, load_config, resolve_seed, write_config,
)
from dyna_dist.data.vehicle_collection import shared_probe
from dyna_dist.inference.posterior import components
from dyna_dist.io.checkpoint import load_model
from dyna_dist.io.common import new_directory, write_json
from dyna_dist.io.dataset import Dataset
from dyna_dist.models.features import prior_moments_and_variance
from dyna_dist.models.gaussian import joint_nll_from_cholesky
from dyna_dist.models.profiles import vehicle_parameters
from dyna_dist.training.trainer import precision


PARAMETER_KEYS = ("friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio")
COVERAGE_Z = {"50": 0.6744897501960817, "90": 1.6448536269514722, "95": 1.959963984540054}


def physical_parameter_samples(dataset, count, rng):
    """Fresh space-filling draws from the declared four-parameter population range."""
    bounds = np.asarray([
        dataset.manifest["config"][name]
        for name in ("friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio")
    ], dtype=np.float64)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("physical sampling requires nonzero range for all four knobs")
    unit = np.column_stack([
        (rng.permutation(count) + rng.random(count)) / count for _ in PARAMETER_KEYS
    ])
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def _models(config, dataset):
    result = {}
    for latent in config.latent_dimensions:
        run = Path(config.training_run) / f"latent-{latent}"
        summary = json.loads((run / "summary.json").read_text())
        split = json.loads((run / "data_split.json").read_text())
        if summary.get("status") != "complete" or summary["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError(f"latent {latent} is incomplete or trained on another dataset")
        if split["fingerprint"] != dataset.fingerprint:
            raise ValueError(f"latent {latent} saved split disagrees with dataset")
        final_model = summary.get("final_model", "best")
        manifest = json.loads((run / final_model / "manifest.json").read_text())
        precision(manifest["model_config"]["dtype"])
        model = load_model(run / final_model)
        if (
            model.config.domain != "st"
            or model.config.latent_dimension != latent
            or model.config.mode != "dynamics"
            or model.config.timestep != dataset.timestep
        ):
            raise ValueError(f"latent {latent} model format does not match vehicle study")
        result[latent] = model
    return result


def _sample_transitions(dataset, config):
    """Select the same speed-stratified true transitions for every BNN."""
    speeds = tuple(float(value) for value in dataset.manifest["config"]["speeds"])
    result = {}
    for member_index, member in enumerate(dataset.instances):
        state_parts, control_parts, next_parts, speed_parts = [], [], [], []
        for speed_index, speed in enumerate(speeds):
            prefix = f"speed-{speed:g}-signal-"
            files = [name for name in member["files"] if Path(name).stem.startswith(prefix)]
            if not files:
                raise ValueError(f"missing {prefix} probes for {member['instance_id']}")
            states, controls, targets = [], [], []
            for filename in files:
                trajectory = dataset.trajectory(filename)
                valid = np.flatnonzero(trajectory["valid"])
                states.append(trajectory["states"][valid])
                controls.append(trajectory["controls"][valid])
                targets.append(trajectory["states"][valid + 1])
            states, controls, targets = map(np.concatenate, (states, controls, targets))
            rng = np.random.default_rng(np.random.SeedSequence([
                config.seed, 101, member_index, speed_index
            ]))
            chosen = rng.choice(
                len(states), size=min(config.one_step_per_speed, len(states)), replace=False
            )
            state_parts.append(states[chosen])
            control_parts.append(controls[chosen])
            next_parts.append(targets[chosen])
            speed_parts.append(np.full(len(chosen), speed))
        states, controls, targets, labels = map(
            np.concatenate, (state_parts, control_parts, next_parts, speed_parts)
        )
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 202, member_index]))
        group_size = min(config.joint_group_size, len(states))
        groups = np.stack([
            rng.choice(len(states), size=group_size, replace=False)
            for _ in range(config.joint_groups)
        ])
        result[member["instance_id"]] = {
            "instance_id": member["instance_id"],
            "split": member["split"],
            "parameters": {key: member[key] for key in PARAMETER_KEYS},
            "states": states,
            "controls": controls,
            "targets": targets,
            "speeds": labels,
            "groups": groups,
        }
    return result


def _errors(prediction, truth):
    error = np.asarray(prediction) - np.asarray(truth)
    for index in (4, 6):
        error[..., index] = (error[..., index] + np.pi) % (2 * np.pi) - np.pi
    return error


def _joint_nll(model, sample):
    dtype = model.normalizer.input_mean.dtype
    prior_mean, prior_factor, variance = prior_moments_and_variance(
        model.parameters, model.config
    )
    values = []
    for indices in sample["groups"]:
        states = jnp.asarray(sample["states"][indices], dtype)
        controls = jnp.asarray(sample["controls"][indices], dtype)
        targets = jnp.asarray(sample["targets"][indices], dtype)
        mean, basis = components(model, states, controls)
        normalized = model.targets(states, controls, targets)
        nll = joint_nll_from_cholesky(
            prior_mean, prior_factor, basis, normalized - mean,
            variance, jnp.ones(len(indices), dtype=dtype),
        ) / (len(indices) * normalized.shape[-1])
        values.append(float(nll))
    return float(np.mean(values))


def _one_step_metrics(models, selected, config, emit):
    rows = []
    for latent, model in models.items():
        posterior = model.prior()
        dtype = model.normalizer.input_mean.dtype
        for sample in selected.values():
            prediction = posterior.predict(
                jnp.asarray(sample["states"], dtype),
                jnp.asarray(sample["controls"], dtype),
            )
            error = _errors(np.asarray(prediction.mean), sample["targets"])
            standard_deviation = np.sqrt(np.maximum(
                np.diagonal(np.asarray(prediction.predictive_covariance), axis1=-2, axis2=-1),
                0,
            ))
            finite = np.isfinite(error).all(axis=1) & np.isfinite(standard_deviation).all(axis=1)
            if not finite.any():
                raise FloatingPointError(f"latent {latent} has no finite one-step predictions")
            nll = _joint_nll(model, sample)
            for speed in np.unique(sample["speeds"]):
                chosen = (sample["speeds"] == speed) & finite
                if not chosen.any():
                    continue
                local_error = error[chosen]
                local_std = standard_deviation[chosen]
                coverage = {
                    level: np.mean(np.abs(local_error) <= z * local_std, axis=0).tolist()
                    for level, z in COVERAGE_Z.items()
                }
                rmse = np.sqrt(np.mean(local_error**2, axis=0))
                rows.append({
                    "latent": latent,
                    "instance_id": sample["instance_id"],
                    "split": sample["split"],
                    "speed_mps": float(speed),
                    "parameters": sample["parameters"],
                    "sample_count": int(chosen.sum()),
                    "nonfinite_fraction": float(1 - finite[sample["speeds"] == speed].mean()),
                    "rmse": rmse.tolist(),
                    "position_rmse_m": float(np.linalg.norm(rmse[:2])),
                    "coverage": coverage,
                    "joint_nll_per_component": nll,
                })
        emit(f"[vehicle-eval] latent {latent}: one-step predictions complete")
    aggregate = []
    for latent in models:
        for split in ("train", "validation", "test"):
            subset = [r for r in rows if r["latent"] == latent and r["split"] == split]
            if not subset:
                continue
            aggregate.append({
                "latent": latent,
                "split": split,
                "instance_count": len({r["instance_id"] for r in subset}),
                "position_rmse_m": float(np.mean([r["position_rmse_m"] for r in subset])),
                "joint_nll_per_component": float(np.mean([
                    next(r["joint_nll_per_component"] for r in subset if r["instance_id"] == i)
                    for i in {r["instance_id"] for r in subset}
                ])),
                "coverage_95_mean_component": float(np.mean([
                    np.mean(r["coverage"]["95"]) for r in subset
                ])),
                "rmse_by_component": np.mean([r["rmse"] for r in subset], axis=0).tolist(),
            })
    return rows, aggregate


def fresh_probe(dataset, config, speed, signal, *, extra_steps=0):
    """New action tape with a common nominal state at each evaluation start."""
    values = dataset.manifest["config"]
    collection = VehicleCollectionConfig(**{
        field.name: values[field.name] for field in fields(VehicleCollectionConfig)
        if field.name in values
    })
    steps = max(collection.steps, max(config.rollout_starts) + config.horizon + extra_steps)
    rng = np.random.default_rng(np.random.SeedSequence([
        config.seed, 404, int(speed * 100), signal,
    ]))
    probe = shared_probe(collection, speed, rng, steps=steps)
    parameters = vehicle_parameters("fullscale", sum(collection.friction) / 2, 1.0, "fixed")
    simulator = GymST(parameters, dataset.timestep, collection.integrator_timestep)
    nominal = _physical_rollouts([simulator], probe["initial_state"], probe["controls"])[0]
    return probe["controls"], nominal


def _simulators(dataset, parameter_values):
    timestep = dataset.timestep
    substep = dataset.manifest["config"]["integrator_timestep"]
    result = []
    for mu, front, rear, mass in parameter_values:
        parameters = vehicle_parameters("fullscale", float(mu), float(mass), "fixed")
        parameters["C_Sf"] *= front
        parameters["C_Sr"] *= rear
        result.append(GymST(parameters, timestep, substep))
    if result[0].source_fingerprint != dataset.manifest["source_fingerprint"]:
        raise ValueError("Gym ST source differs from the collection source")
    return result


def _physical_rollouts(simulators, initial, controls):
    result = np.empty((len(simulators), len(controls) + 1, len(initial)), dtype=np.float64)
    result[:, 0] = initial
    for index, simulator in enumerate(simulators):
        for step, action in enumerate(controls, start=1):
            result[index, step] = simulator.step(result[index, step - 1], action)
    return result


def _invalid_physics(paths):
    state = paths[:, 1:, :]
    return np.any(
        (state[..., 2] < -0.91) | (state[..., 2] > 0.91)
        | (state[..., 3] < -1e-5) | (state[..., 3] > 20.0 + 1e-5)
        | (np.abs(state[..., 5]) > 4.0) | (np.abs(state[..., 6]) > 0.8),
        axis=1,
    )


def trajectory_cover(sample_paths, truth_paths, horizon):
    """Coverage of independent true paths by a sampled function ensemble."""
    paths = np.asarray(sample_paths)[:, :horizon + 1]
    truth = np.asarray(truth_paths)[:, :horizon + 1]
    finite = np.isfinite(paths).all(axis=(1, 2))
    physical_invalid = _invalid_physics(paths[finite])
    if finite.sum() < 4:
        return {
            "finite_samples": int(finite.sum()),
            "nonfinite_fraction": float(1 - finite.mean()),
            "physical_violation_fraction": None,
            "unavailable": "fewer than four finite function trajectories",
        }
    state_center = paths[finite].mean(axis=0)
    for index in (4, 6):
        state_center[:, index] = np.arctan2(
            np.sin(paths[finite, :, index]).mean(axis=0),
            np.cos(paths[finite, :, index]).mean(axis=0),
        )
    state_errors = _errors(state_center[None], truth)[:, 1:]
    state_rmse = np.sqrt(np.mean(state_errors ** 2, axis=(0, 1)))
    terminal_rmse = np.sqrt(np.mean(state_errors[:, -1] ** 2, axis=0))
    paths = paths[finite, :, :2]
    truth = truth[:, :, :2]
    low, high = np.quantile(paths, [0.025, 0.975], axis=0)
    within = (truth >= low[None]) & (truth <= high[None])
    center = paths.mean(axis=0)
    sample_max = np.linalg.norm(paths - center, axis=-1).max(axis=1)
    tube_radius = float(np.quantile(sample_max, 0.95))
    truth_max = np.linalg.norm(truth - center, axis=-1).max(axis=1)
    terminal_radius = float(np.quantile(
        np.linalg.norm(paths[:, -1] - center[-1], axis=-1), 0.95
    ))
    result = {
        "finite_samples": int(len(paths)),
        "nonfinite_fraction": float(1 - finite.mean()),
        "physical_violation_fraction": float(physical_invalid.mean()),
        "pointwise_coordinate_95_coverage": float(within.mean()),
        "pointwise_xy_95_coverage": float(within.all(axis=-1).mean()),
        "whole_path_marginal_95_coverage": float(within.all(axis=(1, 2)).mean()),
        "simultaneous_tube_95_coverage": float((truth_max <= tube_radius).mean()),
        "simultaneous_tube_radius_m": tube_radius,
        "terminal_radius_95_m": terminal_radius,
        "mean_position_rmse_m": float(np.sqrt(np.mean(
            np.sum((center[None, 1:] - truth[:, 1:]) ** 2, axis=-1)
        ))),
        "terminal_mean_error_m": float(np.linalg.norm(center[-1] - truth[:, -1].mean(axis=0))),
        "state_rmse": state_rmse.tolist(),
        "terminal_state_rmse": terminal_rmse.tolist(),
        "terminal_position_rmse_m": float(np.linalg.norm(terminal_rmse[:2])),
    }
    for level in (50, 90, 95):
        alpha = (1 - level / 100) / 2
        low, high = np.quantile(paths, [alpha, 1 - alpha], axis=0)
        within = ((truth >= low) & (truth <= high)).all(axis=-1)
        radius = float(np.quantile(sample_max, level / 100))
        result[f"pointwise_xy_{level}_coverage"] = float(within[:, 1:].mean())
        result[f"simultaneous_tube_{level}_coverage"] = float((truth_max <= radius).mean())
        result[f"mean_xy_{level}_band_width_m"] = float((high - low)[1:].mean())
    return result


def distribution_gap(sample_paths, reference_paths, horizon):
    """Energy and empirical W1 with RMS XY path distance as ground metric."""
    from scipy.spatial.distance import cdist

    a = np.asarray(sample_paths)[:, 1:horizon + 1, :2]
    b = np.asarray(reference_paths)[:, 1:horizon + 1, :2]
    a = a[np.isfinite(a).all(axis=(1, 2))]
    b = b[np.isfinite(b).all(axis=(1, 2))]
    if not len(a) or not len(b):
        return {"mean_path_bias_m": None, "path_energy_distance_m": None,
                "wasserstein1_path_m": None}
    mean_bias = np.sqrt(np.mean(np.sum((a.mean(axis=0) - b.mean(axis=0)) ** 2, axis=-1)))
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    # Euclidean path distance divided by sqrt(H) is an RMS XY separation in meters.
    cross = cdist(a, b) / np.sqrt(horizon)
    energy = 2 * cross.mean() - (cdist(a, a).mean() + cdist(b, b).mean()) / np.sqrt(horizon)
    return {"mean_path_bias_m": float(mean_bias), "path_energy_distance_m": float(energy),
            "wasserstein1_path_m": empirical_wasserstein1(cross)}


def empirical_wasserstein1(cost):
    """Exact transport between equally weighted empirical samples, including unequal counts."""
    from scipy.optimize import linear_sum_assignment, linprog

    cost = np.asarray(cost, dtype=np.float64)
    m, n = cost.shape
    if m == n:
        rows, columns = linear_sum_assignment(cost)
        return float(cost[rows, columns].mean())
    from scipy.sparse import coo_matrix

    entries = np.arange(m * n)
    rows = np.concatenate((np.repeat(np.arange(m), n), m + np.tile(np.arange(n), m)))
    constraints = coo_matrix(
        (np.ones(2 * m * n), (rows, np.tile(entries, 2))), shape=(m + n, m * n)
    ).tocsr()
    transport = linprog(
        cost.ravel(), A_eq=constraints,
        b_eq=np.r_[np.full(m, 1 / m), np.full(n, 1 / n)], bounds=(0, None), method="highs",
    )
    if not transport.success:
        raise RuntimeError(f"Wasserstein transport failed: {transport.message}")
    return float(transport.fun)


def _window_metrics(predictions, real_paths, repeat_paths, truths, config):
    rows = []
    for horizon in config.report_horizons:
        reference = trajectory_cover(real_paths, truths["test"], horizon)
        reference.update(distribution_gap(real_paths, repeat_paths, horizon))
        learned = {}
        for latent, paths in predictions.items():
            cover = trajectory_cover(paths, truths["test"], horizon)
            cover.update(distribution_gap(paths, real_paths, horizon))
            cover["split_coverage"] = {
                split: trajectory_cover(paths, target, horizon) for split, target in truths.items()
            }
            learned[str(latent)] = cover
        rows.append({"horizon": horizon, "real_model_cover": reference, "bnn_cover": learned})
    return rows


def _repeat_parameter_values(dataset, config):
    return physical_parameter_samples(
        dataset, config.model_particles,
        np.random.default_rng(np.random.SeedSequence([config.seed, 505])),
    )


def _rollout_metrics(models, dataset, config, root, emit):
    test_members = dataset.members("test")
    if not test_members:
        raise ValueError("vehicle study needs unseen test physics instances")
    real_values = physical_parameter_samples(
        dataset, config.real_models,
        np.random.default_rng(np.random.SeedSequence([config.seed, 303])),
    )
    simulators = _simulators(dataset, real_values)
    repeat_values = _repeat_parameter_values(dataset, config)
    repeat_simulators = _simulators(dataset, repeat_values)
    split_members = {split: dataset.members(split) for split in ("train", "validation", "test")}
    split_simulators = {
        split: _simulators(dataset, np.array([
            [member[key] for key in PARAMETER_KEYS] for member in members
        ])) for split, members in split_members.items()
    }
    sampled = {
        latent: model.sample_functions(jax.random.key(config.seed), config.model_particles)
        for latent, model in models.items()
    }
    np.savez_compressed(
        root / "sampled_functions.npz",
        real_parameter_values=real_values,
        independent_real_parameter_values=repeat_values,
        **{f"latent_{latent}_coefficients": np.asarray(functions.coefficients)
           for latent, functions in sampled.items()},
    )
    windows = []
    (root / "rollouts").mkdir(exist_ok=True)
    for speed in config.rollout_speeds:
        for signal in config.rollout_signals:
            tape, nominal = fresh_probe(dataset, config, speed, signal)
            for start in config.rollout_starts:
                probe_id = f"speed-{speed:g}-signal-{signal}-start-{start}"
                initial, controls = nominal[start], tape[start:start + config.horizon]
                truths = {split: _physical_rollouts(sims, initial, controls)
                          for split, sims in split_simulators.items()}
                truth = truths["test"]
                real_paths = _physical_rollouts(simulators, initial, controls)
                repeat_paths = _physical_rollouts(repeat_simulators, initial, controls)
                predictions = {}
                for latent, functions in sampled.items():
                    learned = np.asarray(functions.rollout(initial, controls)[0])
                    initial_states = np.broadcast_to(initial, (len(learned), 1, len(initial)))
                    predictions[latent] = np.concatenate((initial_states, learned), axis=1)
                np.savez_compressed(
                    root / "rollouts" / f"{probe_id}.npz",
                    controls=controls, truth=truth, real_model_paths=real_paths,
                    independent_real_model_paths=repeat_paths,
                    **{f"truth_{split}": paths for split, paths in truths.items()},
                    **{f"latent_{latent}_paths": paths for latent, paths in predictions.items()},
                )
                metrics = _window_metrics(predictions, real_paths, repeat_paths, truths, config)
                windows.append({
                    "probe_id": probe_id, "speed_mps": speed, "signal": signal,
                    "start_step": start, "initial_state": initial.tolist(),
                    "test_instance_ids": [member["instance_id"] for member in test_members],
                    "metrics": metrics,
                })
                emit(f"[vehicle-eval] {probe_id}: fresh-control trajectories complete")
    return windows


def evaluate_vehicle_bnn(config, emit=print):
    config.validate()
    config = replace(config, seed=resolve_seed(config.seed))
    dataset = Dataset(config.dataset)
    if dataset.domain.name != "st" or dataset.manifest["config"]["profile"] != "fullscale":
        raise ValueError("vehicle BNN evaluation requires the fullscale ST study dataset")
    if not set(config.rollout_speeds).issubset(set(dataset.manifest["config"]["speeds"])):
        raise ValueError("rollout speeds are not present in the paired dataset")
    models = _models(config, dataset)
    selected = _sample_transitions(dataset, config)
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    one_step_rows, one_step_aggregate = _one_step_metrics(models, selected, config, emit)
    windows = _rollout_metrics(models, dataset, config, root, emit)
    metrics = {
        "schema_version": 3,
        "timestep": dataset.timestep,
        "dataset_fingerprint": dataset.fingerprint,
        "model_fingerprints": {str(latent): model.fingerprint for latent, model in models.items()},
        "parameter_keys": PARAMETER_KEYS,
        "one_step_rows": one_step_rows,
        "one_step_aggregate": one_step_aggregate,
        "rollout_windows": windows,
        "notes": {
            "prior_inference": "stored_mean_factor",
            "training_split": "in-sample physics and recorded transitions; diagnostic only",
            "validation_split": "held-out original physics; used for checkpoint selection",
            "test_split": f"{len(dataset.members('test'))} held-out physics; not used in training or selection",
            "real_model_cover": "fresh Latin-hypercube ST parameter draws within the collection bounds",
            "function_sampling": "one coefficient per BNN function, fixed across every control step and window",
            "pointwise_coverage": "coordinatewise 95% quantile bands are not simultaneous 95% path bands",
            "simultaneous_tube": "95th percentile of sample maximum XY distance from sample mean path",
            "wasserstein": "exact empirical W1; ground cost is RMS future XY path separation in meters",
            "distribution_baseline": "independent ST parameter ensemble with the BNN particle count; seed stream 505",
        },
    }
    if config.mppi_enabled:
        from .vehicle_mppi import evaluate_mppi

        metrics["mppi"] = evaluate_mppi(models, dataset, config, root, windows, emit)
    write_json(root / "metrics.json", metrics)
    from .vehicle_bnn_plots import plot_vehicle_bnn, write_vehicle_bnn_report

    plot_vehicle_bnn(root, config, metrics)
    write_vehicle_bnn_report(root, config, metrics)
    return root


def refresh_vehicle_bnn_evaluation(root, report_horizons=(1, 5, 10, 15), emit=print):
    """Refresh metrics/report from saved predictions; preserve trained models and MPPI results."""
    root = Path(root)
    config = replace(load_config(VehicleBNNEvaluationConfig, root / "config.yaml"),
                     report_horizons=tuple(report_horizons)).validate()
    metrics = json.loads((root / "metrics.json").read_text())
    dataset = Dataset(config.dataset)
    if dataset.fingerprint != metrics["dataset_fingerprint"]:
        raise ValueError("saved evaluation and dataset fingerprints differ")
    with np.load(root / "sampled_functions.npz", allow_pickle=False) as shard:
        samples = {name: shard[name] for name in shard.files}
    if "independent_real_parameter_values" not in samples:
        samples["independent_real_parameter_values"] = _repeat_parameter_values(dataset, config)
        np.savez_compressed(root / "sampled_functions.npz", **samples)
    repeat_simulators = _simulators(dataset, samples["independent_real_parameter_values"])
    for window in metrics["rollout_windows"]:
        filename = root / "rollouts" / f"{window['probe_id']}.npz"
        with np.load(filename, allow_pickle=False) as shard:
            data = {name: shard[name] for name in shard.files}
        if "independent_real_model_paths" not in data:
            data["independent_real_model_paths"] = _physical_rollouts(
                repeat_simulators, np.asarray(window["initial_state"]), data["controls"]
            )
            np.savez_compressed(filename, **data)
        window["metrics"] = _window_metrics(
            {latent: data[f"latent_{latent}_paths"] for latent in config.latent_dimensions},
            data["real_model_paths"], data["independent_real_model_paths"],
            {split: data[f"truth_{split}"] for split in ("train", "validation", "test")}, config,
        )
        emit(f"[vehicle-eval] {window['probe_id']}: coverage, energy, and W1 updated")
    metrics["schema_version"] = 3
    metrics["notes"]["wasserstein"] = "exact empirical W1; ground cost is RMS future XY path separation in meters"
    metrics["notes"]["distribution_baseline"] = "independent ST parameter ensemble with the BNN particle count; seed stream 505"
    write_config(config, root / "config.yaml")
    write_json(root / "metrics.json", metrics)
    from .vehicle_bnn_summary import plot_compact_summary
    from .vehicle_bnn_plots import write_vehicle_bnn_report

    plot_compact_summary(root, config, metrics)
    from .vehicle_bnn_plots import _plot_summary

    _plot_summary(root, config, metrics["rollout_windows"])
    write_vehicle_bnn_report(root, config, metrics)
    return root
