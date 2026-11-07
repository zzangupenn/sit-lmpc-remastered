"""Compare vehicle population sizes on exactly the same held-out rollouts."""

import json
from pathlib import Path

import numpy as np

from dyna_dist.config import VehicleBNNEvaluationConfig, load_config
from dyna_dist.io.common import write_json
from dyna_dist.io.dataset import Dataset


def frozen_test_checks(previous, current):
    datasets = []
    for root in (previous, current):
        config = load_config(VehicleBNNEvaluationConfig, root / "evaluation-option-1/config.yaml")
        datasets.append(Dataset(config.dataset))
    before, after = datasets
    tests = [{member["instance_id"]: member for member in dataset.members("test")}
             for dataset in datasets]
    if tests[0].keys() != tests[1].keys():
        raise ValueError("test physics identities differ")
    count = 0
    for name, original in tests[0].items():
        retained = tests[1][name]
        if original["parameters"] != retained["parameters"] or original["files"] != retained["files"]:
            raise ValueError("test parameters or shard names differ")
        for filename in original["files"]:
            if before.manifest["shard_checksums"][filename] != after.manifest["shard_checksums"][filename]:
                raise ValueError("test shard checksums differ")
            count += 1
    metrics = [json.loads((root / "evaluation-option-1/metrics.json").read_text())
               for root in (previous, current)]
    inference = [item.get("notes", {}).get("prior_inference", "legacy_precision_round_trip")
                 for item in metrics]
    if inference[0] != inference[1]:
        raise ValueError("inference implementations differ; reevaluate the frozen baseline before comparing")
    if [w["probe_id"] for w in metrics[0]["rollout_windows"]] != [
        w["probe_id"] for w in metrics[1]["rollout_windows"]
    ]:
        raise ValueError("evaluation windows differ")
    for option in (1, 2):
        for window in metrics[0]["rollout_windows"]:
            name = f"{window['probe_id']}.npz"
            with np.load(previous / f"evaluation-option-{option}/rollouts" / name) as a, np.load(
                current / f"evaluation-option-{option}/rollouts" / name
            ) as b:
                for field in ("controls", "truth_test", "real_model_paths", "independent_real_model_paths"):
                    np.testing.assert_array_equal(a[field], b[field])
    return {"identical_test_physics": len(tests[0]), "identical_test_shard_checksums": count,
            "identical_rollout_windows": len(metrics[0]["rollout_windows"]),
            "split_counts": [{split: len(dataset.members(split))
                              for split in ("train", "validation", "test")} for dataset in datasets],
            "identical_controls_test_truths_and_physical_reference_paths": True}


def compare_dataset_sizes(previous, current):
    """Write a concise comparison without fitting/selecting models on test data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    previous, current = Path(previous).resolve(), Path(current).resolve()
    plans = [json.loads((root / "plan.json").read_text()) for root in (previous, current)]
    for key in ("latents", "steps", "seed", "evaluation_seed", "mppi"):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"dataset comparison must match {key}")
    checks = frozen_test_checks(previous, current)
    counts = checks["split_counts"]
    comparisons = [json.loads((root / "comparison.json").read_text())["rows"]
                   for root in (previous, current)]
    dimensions = plans[1]["latents"]
    fields = ("terminal_position_rmse_m", "simultaneous_tube_95_coverage", "path_energy_distance_m",
              "wasserstein1_path_m", "simultaneous_tube_radius_m")
    rows = []
    for old in comparisons[0]:
        new = next(row for row in comparisons[1] if all(row[key] == old[key]
                   for key in ("option", "latent", "horizon")))
        rows.append({**{key: old[key] for key in ("option", "latent", "horizon")},
                     "previous": {key: old[key] for key in fields},
                     "current": {key: new[key] for key in fields}})
    write_json(current / "dataset_size_comparison.json", {
        "previous_study": str(previous), "current_study": str(current),
        "checks": checks, "rows": rows,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    labels = ("15-step endpoint XY RMSE [m] ↓", "15-step path Wasserstein-1 [m] ↓",
              "95% whole-path coverage", "95% whole-path radius [m]")
    for axis, field, label in zip(axes.flat, (fields[0], fields[3], fields[1], fields[4]), labels, strict=True):
        for option, color in ((1, "tab:blue"), (2, "tab:orange")):
            for index, style in enumerate(("--", "-")):
                label_size = f"{counts[index]['train']}/{counts[index]['validation']}"
                values = [next(row for row in rows if (row["option"], row["latent"], row["horizon"])
                               == (option, latent, 15))["previous" if index == 0 else "current"][field]
                          for latent in dimensions]
                axis.plot(dimensions, values, style, marker="o", color=color,
                          label=f"Option {option}, {label_size} train/val")
        axis.set(xlabel="Latent dimension", ylabel=label, xticks=dimensions)
        axis.grid(alpha=0.2)
        if "coverage" in field:
            axis.axhline(0.95, ls=":", color="0.4", lw=1)
            axis.set_ylim(0, 1.03)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"More training physics, same {checks['identical_test_physics']} test vehicles "
                 f"and evaluation controls; {plans[1]['steps']:,} updates")
    fig.tight_layout()
    figure = current / "plots/dataset_size_effect.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    configuration = load_config(VehicleBNNEvaluationConfig, current / "evaluation-option-1/config.yaml")
    dataset = Dataset(configuration.dataset)
    per_instance = dataset.manifest["transitions"] // len(dataset.instances)
    transition_draws = 4 * 64 * plans[1]["steps"]
    sampling_note = (
        f" With four vehicles × 64 transitions × {plans[1]['steps']:,} updates, the neural "
        f"M-step/gradient sampler makes {transition_draws / 1e6:.2f} million transition draws: "
        f"about {transition_draws / (counts[0]['train'] * per_instance):.2f} dataset passes before "
        f"versus {transition_draws / (counts[1]['train'] * per_instance):.2f} now "
        "(sampling with replacement, not exhaustive epochs)."
    ) if min(counts[0]["train"], counts[1]["train"], per_instance) > 0 else ""
    lines = [f"# More training physics: {counts[0]['train']} → {counts[1]['train']} combinations", "",
             f"New split: **{counts[1]['train']} train / {counts[1]['validation']} validation / "
             f"{counts[1]['test']} unchanged test** physics, {per_instance:,} transitions "
             f"per combination ({dataset.manifest['transitions']:,} total). All "
             f"{counts[1]['train'] + counts[1]['validation']} training/validation combinations are new "
             "Latin-hypercube draws inside the previous friction, stiffness and mass ranges. "
             "This compares two population designs, not a nested-data ablation: the original 27 training "
             "and six validation combinations were replaced, and the validation split also changed.", "",
             f"Verified: all {checks['identical_test_physics']} test parameter sets and "
             f"{checks['identical_test_shard_checksums']:,} shard checksums are unchanged. All "
             f"{checks['identical_rollout_windows']} fresh-control windows have byte-identical numeric "
             "controls, test GT and physical reference paths across studies, for both options. "
             "Both retain latent 4/8/16/32, training seed 20261018, evaluation seed 20261019, "
             "20,000 neural updates, prior refit for option 1, and full-instance EM for option 2. "
             "Training normalizers are refitted using only each dataset's training physics.", "",
             "## Test results at 15 steps (1.5 seconds)", "",
             "Each cell shows **previous → new**. Error, energy and W1: lower is better. Coverage "
             "must be judged together with distribution distance and width, not maximized alone.", "",
             "| Option | Latent | Endpoint RMSE [m] | 95% path coverage | Energy [m] | W1 [m] | Path radius [m] |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["horizon"] != 15:
            continue
        cells = [f"{row['option']}", f"{row['latent']}"]
        for field in fields:
            fmt = ".1%" if "coverage" in field else ".4f"
            cells.append(f"{row['previous'][field]:{fmt}} → {row['current'][field]:{fmt}}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "![Effect of increasing population size](plots/dataset_size_effect.png)", "",
              "## Interpretation", ""]
    for option in (1, 2):
        selected = [row for row in rows if row["option"] == option and row["horizon"] == 15]
        wins = {field: sum(row["current"][field] < row["previous"][field] for row in selected)
                for field in (fields[0], fields[3])}
        best = min(selected, key=lambda row: row["current"][fields[3]])
        lines.append(f"- Option {option}: lower endpoint RMSE at {wins[fields[0]]}/{len(dimensions)} "
                     f"latent sizes; lower W1 at {wins[fields[3]]}/{len(dimensions)}. Its lowest new W1 "
                     f"is latent {best['latent']}: {best['current'][fields[3]]:.4f} m, with "
                     f"{best['current'][fields[1]]:.1%} path coverage.")
    terminal = [row for row in rows if row["horizon"] == 15]
    leader = min(terminal, key=lambda row: row["current"][fields[3]])
    within_wins = {field: sum(
        next(row for row in terminal if (row["option"], row["latent"]) == (1, latent))["current"][field]
        < next(row for row in terminal if (row["option"], row["latent"]) == (2, latent))["current"][field]
        for latent in dimensions
    ) for field in (fields[0], fields[3])}
    lines += ["", f"**On the new dataset**, option 1 beats option 2 in endpoint RMSE at "
              f"{within_wins[fields[0]]}/{len(dimensions)} latent sizes and W1 at "
              f"{within_wins[fields[3]]}/{len(dimensions)}. The lowest observed W1 is option "
              f"{leader['option']}, latent {leader['latent']}: RMSE "
              f"{leader['current'][fields[0]]:.4f} m, W1 {leader['current'][fields[3]]:.4f} m, "
              f"coverage {leader['current'][fields[1]]:.1%}. More physics diversity is not an "
              "automatic guarantee of improvement; compare both methods per latent size."]
    metrics = json.loads((current / "evaluation-option-1/metrics.json").read_text())
    reference = [next(row for row in window["metrics"] if row["horizon"] == 15)["real_model_cover"]
                 for window in metrics["rollout_windows"]]
    reference_w1 = np.mean([row["wasserstein1_path_m"] for row in reference])
    reference_coverage = np.mean([row["simultaneous_tube_95_coverage"] for row in reference])
    lines += ["", f"For scale, two independent 256-vehicle ST samples have W1 {reference_w1:.4f} m "
              f"and {reference_coverage:.1%} coverage on the same test cases. This is a finite-sample "
              "reference discrepancy, not a mathematical lower bound. Near-95% aggregate coverage "
              "alone does not establish a matched distribution or calibrated rare-event probabilities."]
    lines += ["", f"One seed per model; no statistical-significance claim. Keeping {plans[1]['steps']:,} updates fixes "
              "the neural-update budget, not epochs: each new training vehicle receives fewer sampled "
              "neural updates on average." + sampling_note + " Option 2 still processes every transition in each E-step, "
              "so its total work increases with dataset size. Differences therefore include population "
             "sampling, validation selection, normalization and optimization, not just transition count.", "",
              "The RMSE is for the prior-ensemble mean against individual vehicles; it includes true "
              "between-vehicle variation. W1/energy compare sampled function rollouts against a fresh "
              "physical-model population. Nominal 95% path tubes are empirical, not safety certificates.", "",
              "[New complete training/evaluation report, curves and 1/5/10/15-step metrics](REPORT.md) · "
              "[All trajectory sections](TRAJECTORY_SECTIONS.md) · "
              "[All old/new horizon metrics and matching checks](dataset_size_comparison.json) · "
              f"[Previous study]({previous / 'REPORT.md'}).", ""]
    output = current / "DATASET_SIZE_COMPARISON.md"
    from .trajectory_sections import sampled_trajectory_markdown

    section_index = lines.index("## Interpretation")
    lines[section_index:section_index] = sampled_trajectory_markdown(current)
    output.write_text("\n".join(lines))
    return output
