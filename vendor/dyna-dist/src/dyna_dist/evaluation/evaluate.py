"""No metadata leakage: adapt from transitions, compare with held-out Gym truth."""

from dataclasses import replace
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from dyna_dist.config import resolve_seed, write_config
from dyna_dist.data.transforms import wrap
from dyna_dist.inference.posterior import components
from dyna_dist.io.checkpoint import load_model
from dyna_dist.io.common import new_directory, write_json
from dyna_dist.io.dataset import Dataset
from dyna_dist.models.features import prior_and_variance
from dyna_dist.models.gaussian import joint_nll
from dyna_dist.models.st import step as nominal_step
from dyna_dist.training.trainer import precision


def rollout_window(dataset, member, excluded_ids, horizon):
    for filename in member["files"]:
        trajectory = dataset.trajectory(filename)
        all_ids = dataset.transition_ids(member, filename, trajectory)
        for start in range(len(trajectory["controls"]) - horizon + 1):
            ids = set(all_ids[start : start + horizon])
            if ids.isdisjoint(excluded_ids) and trajectory["valid"][start : start + horizon].all():
                return (
                    trajectory["states"][start],
                    trajectory["controls"][start : start + horizon],
                    trajectory["states"][start + 1 : start + horizon + 1],
                )
    return None


def _errors(prediction, truth, domain="st"):
    error = np.asarray(prediction) - np.asarray(truth)
    if domain != "st":
        return error
    error[..., 4] = np.asarray(wrap(jnp.asarray(error[..., 4])))
    error[..., 6] = np.asarray(wrap(jnp.asarray(error[..., 6])))
    return error


def evaluate(config, emit=print):
    config.validate()
    config = replace(config, seed=resolve_seed(config.seed))
    model_manifest = json.loads((Path(config.model_path) / "manifest.json").read_text())
    precision(model_manifest["model_config"]["dtype"])
    model = load_model(config.model_path)
    dataset = Dataset(config.dataset)
    if not np.isclose(dataset.timestep, model.config.timestep):
        raise ValueError("evaluation dataset and model intervals differ")
    if dataset.domain.name != model.config.domain:
        raise ValueError("evaluation dataset and model domains differ")
    if (
        model.config.domain == "st"
        and dataset.manifest["config"]["profile"] != model.config.baseline_profile
    ):
        raise ValueError("evaluation vehicle profile differs from trained profile")
    members = dataset.members(config.split)
    if not members:
        raise ValueError(f"no instances in evaluation split {config.split}")
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    rows, plot_data = [], None
    for member_index, member in enumerate(members):
        data = dataset.transitions(member)
        valid_sizes = [size for size in config.context_sizes if size < len(data)]
        if not valid_sizes:
            continue
        reserved = max(valid_sizes)
        query = data.take(np.arange(reserved, len(data)))
        window = rollout_window(dataset, member, frozenset(data.ids[:reserved]), config.horizon)
        prior_prediction = model.prior().predict(query.states, query.controls)
        prior_error = _errors(prior_prediction.mean, query.next_states, model.config.domain)
        for context_size in config.context_sizes:
            if len(data) <= context_size:
                continue
            posterior = model.prior()
            if config.conditioning_mode == "context":
                posterior = posterior.condition(data.take(np.arange(context_size)))
            prediction = posterior.predict(query.states, query.controls)
            error = _errors(prediction.mean, query.next_states, model.config.domain)
            b, basis = components(
                model,
                jnp.asarray(query.states, dtype=prediction.mean.dtype),
                jnp.asarray(query.controls, dtype=prediction.mean.dtype),
            )
            targets = model.targets(
                jnp.asarray(query.states, dtype=prediction.mean.dtype),
                jnp.asarray(query.controls, dtype=prediction.mean.dtype),
                jnp.asarray(query.next_states, dtype=prediction.mean.dtype),
            )
            _, variance = prior_and_variance(model.parameters, model.config)
            nll = float(
                joint_nll(
                    posterior.state,
                    basis,
                    targets - b,
                    variance,
                    jnp.ones(len(query), dtype=targets.dtype),
                )
            ) / (model.domain.state_dimension * len(query))
            std = np.sqrt(
                np.diagonal(np.asarray(prediction.predictive_covariance), axis1=-2, axis2=-1)
            )
            epi_std = np.sqrt(
                np.maximum(
                    np.diagonal(np.asarray(prediction.epistemic_covariance), axis1=-2, axis2=-1), 0
                )
            )
            finite = bool(np.isfinite(error).all() and np.isfinite(std).all())
            row = {
                "instance_id": member["instance_id"],
                "context_size": context_size,
                "query_count": len(query),
                "one_step_finite": finite,
                "joint_nll_normalized_per_component": nll if np.isfinite(nll) else None,
                "joint_nll_physical_per_component": nll
                + float(jnp.log(model.normalizer.target_scale).mean())
                if np.isfinite(nll)
                else None,
                "prior_mean_rmse": np.sqrt(np.mean(prior_error**2, axis=0)).tolist(),
                "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist() if finite else None,
                "coverage_95_predictive": np.mean(np.abs(error) <= 1.96 * std, axis=0).tolist()
                if finite
                else None,
                "mean_epistemic_std": epi_std.mean(axis=0).tolist()
                if np.isfinite(epi_std).all()
                else None,
            }
            if model.config.domain == "st":
                nominal = jax.vmap(
                    lambda x, u: nominal_step(
                        x,
                        u,
                        model.baseline_parameters,
                        model.config.timestep,
                        model.config.integrator_timestep,
                    )
                )(
                    jnp.asarray(query.states, dtype=prediction.mean.dtype),
                    jnp.asarray(query.controls, dtype=prediction.mean.dtype),
                )
                nominal_error = _errors(nominal, query.next_states)
                row["nominal_st_rmse"] = np.sqrt(np.mean(nominal_error**2, axis=0)).tolist()
            if window is not None:
                initial, controls, truth = window
                # Common base draws and common query/control windows for fair context comparisons.
                key = jax.random.fold_in(jax.random.key(config.seed), member_index)
                samples = posterior.sample_functions(key, config.particles)
                paths = np.asarray(samples.rollout(initial, controls)[0])
                path_finite = np.isfinite(paths).all(axis=(1, 2))
                row["rollout_nonfinite_fraction"] = float(1 - path_finite.mean())
                if path_finite.all():
                    path_error = _errors(paths, truth, model.config.domain)
                    row["rollout_rmse"] = np.sqrt(np.mean(path_error**2, axis=(0, 1))).tolist()
                    low, high = np.quantile(paths[..., :2], [0.025, 0.975], axis=0)
                    row["rollout_position_coverage_95"] = float(
                        np.mean((truth[:, :2] >= low) & (truth[:, :2] <= high))
                    )
                else:
                    row["rollout_rmse"] = None
                    row["rollout_position_coverage_95"] = None
                if plot_data is None and context_size == max(config.context_sizes):
                    plot_data = initial, controls, truth, paths, row
            else:
                row["rollout_unavailable"] = "no disjoint complete window"
            rows.append(row)
            emit(
                f"[evaluate] {member['instance_id']} | context={context_size} | queries={len(query)} "
                f"| joint NLL={nll:.5g}"
            )
    aggregate = []
    for size in config.context_sizes:
        subset = [row for row in rows if row["context_size"] == size]
        finite_rows = [row for row in subset if row["rmse"] is not None]
        if finite_rows:
            count = sum(row["query_count"] for row in finite_rows)
            rmse = np.sqrt(
                sum(np.asarray(row["rmse"]) ** 2 * row["query_count"] for row in finite_rows)
                / count
            )
            coverage = (
                sum(
                    np.asarray(row["coverage_95_predictive"]) * row["query_count"]
                    for row in finite_rows
                )
                / count
            )
            summary = {
                "context_size": size,
                "instances": len(subset),
                "query_count": count,
                "nonfinite_instances": len(subset) - len(finite_rows),
                "rmse": rmse.tolist(),
                "position_rmse": float(np.linalg.norm(rmse[:2])),
                "coverage_95_predictive": coverage.tolist(),
            }
            aggregate.append(summary)
            rate_index = 5 if model.config.domain == "st" else 9
            rate_label = "yaw-rate" if model.config.domain == "st" else "root-speed"
            emit(
                f"[evaluate_summary] context={size} | position RMSE={summary['position_rmse']:.5g} m "
                f"| {rate_label} RMSE={rmse[rate_index]:.5g}"
            )
    write_json(
        root / "metrics.json",
        {
            "schema_version": 1,
            "split": config.split,
            "conditioning_mode": config.conditioning_mode,
            "model_fingerprint": model.fingerprint,
            "domain": model.domain.metadata(),
            "rows": rows,
            "aggregate": aggregate,
            "query_policy": "same query set and rollout window across context sizes",
            "interval_note": "one-step predictive covariance uses local angular coordinates; "
            "rollout samples exclude observation/actuator noise",
        },
    )
    if config.plots:
        from .plots import plot_evaluation

        plot_evaluation(rows, plot_data, root, model.config.domain)
    return root
