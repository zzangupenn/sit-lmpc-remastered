"""Compare completed latent-size sweeps without resampling trained models."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from dyna_dist.config import TrainingConfig, VehicleBNNEvaluationConfig, load_config
from dyna_dist.io.common import write_json
from .dataset_sizes import frozen_test_checks
from .trajectory_sections import plot_trajectory_sections, sampled_trajectory_markdown


def validate_sweeps(baseline, extension):
    """Require the same data, update budget and settings, except latent dimension."""
    roots = [Path(baseline).resolve(), Path(extension).resolve()]
    plans = [json.loads((root / "plan.json").read_text()) for root in roots]
    for key in ("dataset_fingerprint", "steps", "seed", "evaluation_seed", "mppi"):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"latent sweep must keep the same {key}")
    if any(plan["status"] != "complete" for plan in plans):
        raise ValueError("both latent sweeps must be complete")
    if set(plans[0]["latents"]) & set(plans[1]["latents"]):
        raise ValueError("latent sweeps overlap; refusing ambiguous duplicate results")
    for option in (1, 2):
        reference = None
        eval_reference = None
        for root, plan in zip(roots, plans, strict=True):
            model_source = Path(plan.get("model_source_root", root))
            evaluation = asdict(load_config(VehicleBNNEvaluationConfig,
                                             root / f"evaluation-option-{option}/config.yaml"))
            for key in ("training_run", "output", "latent_dimensions"):
                evaluation.pop(key)
            if eval_reference is not None and evaluation != eval_reference:
                raise ValueError("latent sweep evaluation settings differ")
            eval_reference = evaluation
            for latent in plan["latents"]:
                run = model_source / f"option-{option}/latent-{latent}"
                settings = asdict(load_config(TrainingConfig, run / "config.yaml"))
                settings.pop("output")
                if settings["model"].pop("latent_dimension") != latent:
                    raise ValueError("latent model configuration disagrees with its directory")
                if reference is not None and settings != reference:
                    raise ValueError("latent sweep training settings differ beyond latent dimension")
                reference = settings
                summary = json.loads((run / "summary.json").read_text())
                if (summary["status"] != "complete" or summary["steps"] != plan["steps"]
                        or summary["dataset_fingerprint"] != plan["dataset_fingerprint"]):
                    raise ValueError("latent model training is incomplete or mismatched")
    return roots, plans


def _plots(output, rows, dimensions, mppi_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    fields = (("terminal_position_rmse_m", "15-step endpoint XY RMSE [m] ↓"),
              ("wasserstein1_path_m", "15-step path Wasserstein-1 [m] ↓"),
              ("simultaneous_tube_95_coverage", "95% whole-path coverage"),
              ("simultaneous_tube_radius_m", "95% whole-path radius [m]"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        for option, color in ((1, "#0072B2"), (2, "#D55E00")):
            values = [next(row[field] for row in rows if (row["option"], row["latent"], row["horizon"])
                           == (option, latent, 15)) for latent in dimensions]
            axis.plot(dimensions, values, "o-", color=color, label=f"Option {option}")
        axis.set(xlabel="Latent dimension", ylabel=label, xscale="log", xticks=dimensions)
        axis.set_xticklabels([str(value) for value in dimensions])
        if "coverage" in field:
            axis.axhline(0.95, ls="--", color="0.4", lw=1)
            axis.set_ylim(0, 1.03)
        axis.axvline(32, ls=":", color="0.5", lw=0.8)
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Same 100/10/16 physics, seeds and 20k neural updates; dotted line: latent 32")
    fig.tight_layout()
    fig.savefig(directory / "metrics_vs_latent.png", dpi=150)
    plt.close(fig)
    horizons = (1, 5, 10, 15)
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for column, option in enumerate((1, 2)):
        for index, latent in enumerate(dimensions):
            for row_index, (field, label) in enumerate((fields[0], fields[1])):
                values = [next(row[field] for row in rows if
                               (row["option"], row["latent"], row["horizon"]) == (option, latent, horizon))
                          for horizon in horizons]
                axes[row_index, column].plot(horizons, values, "o-", color=palette(index),
                                             lw=2.3 if latent == 32 else 1.2, label=f"latent {latent}")
                axes[row_index, column].set(title=f"Option {option}", xlabel="Prediction steps",
                                            ylabel=label.replace("15-step ", ""), xticks=horizons)
                axes[row_index, column].grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(directory / "short_horizon_comparison.png", dpi=150)
    plt.close(fig)
    if not mppi_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for option, color in ((1, "#0072B2"), (2, "#D55E00")):
        ordered = [next(row for row in mppi_rows if (row["option"], row["latent"]) == (option, latent))
                   for latent in dimensions]
        axes[0].plot(dimensions, [row["probability_mae_1m"] for row in ordered], "o-", color=color,
                     label=f"Option {option}")
        axes[1].plot(dimensions, [row["prediction_ms_64_particles"] for row in ordered], "o-", color=color)
    for axis in axes:
        axis.set(xscale="log", xticks=dimensions, xlabel="Latent dimension")
        axis.set_xticklabels([str(value) for value in dimensions])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("MPPI corridor-exit probability MAE (1 m) ↓")
    axes[1].set_ylabel("64 functions × 1024 candidates × 15 steps [ms]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(directory / "mppi_comparison.png", dpi=150)
    plt.close(fig)


def compare_latent_sweeps(baseline, extension):
    roots, plans = validate_sweeps(baseline, extension)
    checks = frozen_test_checks(*roots)
    output = roots[1] / "latent-sweep"
    output.mkdir(exist_ok=True)
    metrics = {(root, option): json.loads((root / f"evaluation-option-{option}/metrics.json").read_text())
               for root in roots for option in (1, 2)}
    sources = {latent: root for root, plan in zip(roots, plans, strict=True) for latent in plan["latents"]}
    dimensions = sorted(sources)
    rows, mppi_rows = [], []
    for root, plan in zip(roots, plans, strict=True):
        comparison = json.loads((root / "comparison.json").read_text())
        rows.extend({**row, "source_study": str(root),
                     "training_wall_seconds": plan["timings"][f"option-{row['option']}-latent-{row['latent']}"]}
                    for row in comparison["rows"])
        for option in (1, 2):
            evaluation = load_config(VehicleBNNEvaluationConfig, root / f"evaluation-option-{option}/config.yaml")
            mppi = metrics[root, option].get("mppi")
            if mppi is None:
                continue
            width = evaluation.mppi_half_widths.index(1.0)
            for latent in plan["latents"]:
                selected = [row for row in mppi["rows"] if row["latent"] == latent]
                benchmark = next(row for row in mppi["benchmark"]["rows"]
                                 if row["latent"] == latent and row["particles"] == 64)
                mppi_rows.append({"option": option, "latent": latent,
                                  "probability_mae_1m": float(np.mean([
                                      row["probability_mae_by_width"][width] for row in selected])),
                                  "candidate_cost_gap": float(np.mean([
                                      row["selected_candidate_cost_gap"] for row in selected])),
                                  "prediction_ms_64_particles": benchmark["steady_state_ms"]})
    write_json(output / "comparison.json", {"baseline": str(roots[0]), "extension": str(roots[1]),
                                            "matching_checks": checks, "rows": rows, "mppi": mppi_rows})
    _plots(output, rows, dimensions, mppi_rows)
    # Retain a readable five-column comparison rather than squeezing all eight
    # latent sizes into one 16-panel figure. The original four-size gallery stays intact.
    display_dimensions = sorted({32, *plans[1]["latents"]} & set(dimensions))
    plot_trajectory_sections(output, display_dimensions,
                             {1: metrics[roots[1], 1], 2: metrics[roots[1], 2]},
                             source_roots=sources,
                             evaluation_config=roots[1] / "evaluation-option-1/config.yaml")
    lines = ["# Vehicle BNN: extended latent-size comparison", "",
             f"Latents {', '.join(map(str, dimensions))}; both training options, identical "
             f"100/10/16 physics split and {plans[1]['steps']:,} neural updates. Training seed "
             f"{plans[1]['seed']}, evaluation seed {plans[1]['evaluation_seed']}. "
             "The new 48/64/96/128 models were trained from scratch, not expanded from latent 32. "
             "Option 1 includes its train-only Gaussian-prior refit; option 2 retains full-data EM. "
             "No training settings changed beyond latent size; initial random-array shapes necessarily differ.", "",
             f"Matching checks passed: same dataset fingerprint, all {checks['identical_test_physics']} test "
             f"vehicles and {checks['identical_test_shard_checksums']:,} shard checksums; all "
             f"{checks['identical_rollout_windows']} controls, test GT and physical reference paths "
             "match numerically across sweeps. All saved old predictions and models are preserved.", "",
             "## Results at 15 steps (1.5 seconds)", "",
             "| Option | Latent | Test endpoint RMSE [m] ↓ | Energy [m] ↓ | W1 [m] ↓ | 95% path coverage | Path radius [m] | Train [s] |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if plans[0].get("inference"):
        lines[4:4] = ["The baseline models were reevaluated from frozen weights using the stored "
                      "Gaussian mean/factor directly, as for every new model. This avoids a lossy "
                      "float32 covariance/precision round trip; old evaluation files are preserved "
                      "but are not the baseline numbers used here. No baseline model was retrained. "
                      "[Inference audit](../inference-audit/README.md).", ""]
    terminal = sorted([row for row in rows if row["horizon"] == 15], key=lambda row: (row["option"], row["latent"]))
    for row in terminal:
        lines.append(f"| {row['option']} | {row['latent']} | {row['terminal_position_rmse_m']:.4f} | "
                     f"{row['path_energy_distance_m']:.4f} | {row['wasserstein1_path_m']:.4f} | "
                     f"{row['simultaneous_tube_95_coverage']:.1%} | {row['simultaneous_tube_radius_m']:.4f} | "
                     f"{row['training_wall_seconds']:.1f} |")
    reference = [next(row for row in window["metrics"] if row["horizon"] == 15)["real_model_cover"]
                 for window in metrics[roots[1], 1]["rollout_windows"] if window.get("metrics")]
    if reference:
        lines += ["", f"For scale, the independent ST-versus-ST reference has mean W1 "
                  f"{np.mean([row['wasserstein1_path_m'] for row in reference]):.4f} m, energy "
                  f"{np.mean([row['path_energy_distance_m'] for row in reference]):.6f} m and "
                  f"{np.mean([row['simultaneous_tube_95_coverage'] for row in reference]):.1%} "
                  "path coverage. This finite-sample reference is not a mathematical lower bound "
                  "or a safety certificate."]
    lines += ["", "![Accuracy, distribution match, coverage and width](plots/metrics_vs_latent.png)", "",
              "## Interpretation", ""]
    for option in (1, 2):
        selected = [row for row in terminal if row["option"] == option]
        winner = min(selected, key=lambda row: row["wasserstein1_path_m"])
        reference = next(row for row in selected if row["latent"] == 32)
        new = [row for row in selected if row["latent"] in plans[1]["latents"]]
        wins = sum(row["wasserstein1_path_m"] < reference["wasserstein1_path_m"] for row in new)
        lines.append(f"- Option {option}: lowest observed 15-step W1 is latent {winner['latent']} "
                     f"({winner['wasserstein1_path_m']:.4f} m; RMSE {winner['terminal_position_rmse_m']:.4f} m, "
                     f"coverage {winner['simultaneous_tube_95_coverage']:.1%}). "
                     f"{wins}/{len(new)} larger models beat its latent-32 W1.")
    overall = min(terminal, key=lambda row: row["wasserstein1_path_m"])
    previous = next(row for row in terminal if (row["option"], row["latent"]) == (overall["option"], 32))
    improvement = 100 * (1 - overall["wasserstein1_path_m"] / previous["wasserstein1_path_m"])
    lines += ["", f"**Lowest observed distribution error:** option {overall['option']}, latent "
              f"{overall['latent']}; its 15-step W1 is {improvement:.1f}% below that option's "
              "refreshed latent-32 baseline. Larger latent size is not monotonically better; "
              "read coverage together with width, distance and inference cost."]
    if mppi_rows:
        winner_time = next(row["prediction_ms_64_particles"] for row in mppi_rows
                           if (row["option"], row["latent"]) == (overall["option"], overall["latent"]))
        baseline_time = next(row["prediction_ms_64_particles"] for row in mppi_rows
                             if (row["option"], row["latent"]) == (overall["option"], 32))
        lines += ["", f"The measured 64-function prediction time is {winner_time:.1f} ms versus "
                  f"{baseline_time:.1f} ms at latent 32 ({winner_time/baseline_time:.2f}×). "
                  "This is the current sampler implementation's steady-state benchmark, not a "
                  "full-controller rate or an intrinsic lower bound on inference cost."]
    lines += ["", "![Error and W1 at all four horizons](plots/short_horizon_comparison.png)", "",
              "Coverage should approach 95% without excessive width; maximizing coverage is not the goal. "
              "RMSE is for the prior-ensemble mean against individual test vehicles and includes true "
              "between-vehicle variation. Energy and W1 compare coherent function-rollout distributions "
              "with independent physical ST samples, excluding observation noise.", "",
              "This is a single-seed, fixed-update comparison, not equal compute or a convergence test. "
              "There are only 100 independent training physics, even though they provide five million "
              "transitions. Latent dimension 128 exceeds that instance count; the Gaussian population "
              "update retains within-posterior covariance rather than relying on rank-deficient point "
              "estimates alone. Do not interpret a larger latent as automatically better calibrated.", ""]
    if (roots[1] / "performance-audit/README.md").is_file():
        lines += ["Option 2's new runs use a mathematically equivalent pairwise covariance "
                  "reparameterization (cubic rather than quartic cost). Historical small-latent runs "
                  "used the slower direct contraction. Float64 summation order differs, not the "
                  "objective or update equations; cross-sweep wall times also reflect this "
                  "implementation optimization. The actual latent-48 prefix comparison confirms "
                  "that the nonconvex training trace is not bitwise identical, so option-2 "
                  "cross-sweep results are not a numerically isolated latent-size ablation. "
                  "Option-1 results are unaffected. The first partial latent-48 EM run was excluded "
                  "and all four new EM models were restarted from scratch. "
                  "[Benchmark and preserved partial run](../performance-audit/README.md).", ""]
    if mppi_rows:
        lines += ["## MPPI diagnostics", "",
                  "![Corridor probability error and prediction runtime](plots/mppi_comparison.png)", "",
                  "Runtime is the steady-state prediction benchmark, not the complete controller. "
                  "Training times include loading, compilation, validation and refit/EM. Raw MPPI "
                  "candidate-ranking, cost gaps, corridor probabilities and replanning metrics remain "
                  "in each sweep's detailed evaluation reports.", ""]
    lines += sampled_trajectory_markdown(output)
    lines += [f"[Earlier 4/8/16/32 study]({roots[0] / 'REPORT.md'}) · "
              "[New 48/64/96/128 study](../REPORT.md) · "
              "[All metrics and checks](comparison.json) · [All trajectory sections](TRAJECTORY_SECTIONS.md).", ""]
    report = output / "REPORT.md"
    report.write_text("\n".join(lines))
    return report
