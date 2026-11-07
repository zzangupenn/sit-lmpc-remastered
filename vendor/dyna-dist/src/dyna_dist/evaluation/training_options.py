"""Paired comparison report for the two vehicle population-training options."""

import json
from pathlib import Path
import shutil
import numpy as np

from dyna_dist.io.common import write_json


def average(metrics, latent, horizon, field, split="test"):
    values = []
    for window in metrics["rollout_windows"]:
        cover = next(row for row in window["metrics"] if row["horizon"] == horizon)["bnn_cover"][str(latent)]
        if field in cover.get("split_coverage", {}).get(split, {}):
            cover = cover["split_coverage"][split]
        values.append(cover[field])
    return float(np.mean(values))


def pairing_checks(root, dimensions, metrics):
    if metrics[1]["dataset_fingerprint"] != metrics[2]["dataset_fingerprint"]:
        raise ValueError("comparison datasets differ")
    pairs = []
    for latent in dimensions:
        manifests = [json.loads((root / f"option-{option}/latent-{latent}/initial/manifest.json").read_text())
                     for option in (1, 2)]
        if manifests[0]["fingerprint"] != manifests[1]["fingerprint"]:
            raise ValueError(f"latent {latent}: initial models/normalizers differ")
        pairs.append({"latent": latent, "identical_initial_model": True,
                      "initial_fingerprint": manifests[0]["fingerprint"]})
    with np.load(root / "evaluation-option-1/sampled_functions.npz") as a, np.load(
        root / "evaluation-option-2/sampled_functions.npz"
    ) as b:
        for field in ("real_parameter_values", "independent_real_parameter_values"):
            np.testing.assert_array_equal(a[field], b[field])
    for window in metrics[1]["rollout_windows"]:
        name = f"{window['probe_id']}.npz"
        with np.load(root / "evaluation-option-1/rollouts" / name) as a, np.load(
            root / "evaluation-option-2/rollouts" / name
        ) as b:
            for field in ("controls", "truth_train", "truth_validation", "truth_test", "real_model_paths"):
                np.testing.assert_array_equal(a[field], b[field])
    return {"same_dataset": True, "same_physical_samples_controls_and_truths": True,
            "matched_initial_models": pairs, "paired_windows": len(metrics[1]["rollout_windows"])}


def plots(root, dimensions, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [1, 5, 10, 15]
    output = root / "plots"
    output.mkdir(exist_ok=True)
    for filename, fields, labels in (
        ("error_coverage", ("terminal_position_rmse_m", "simultaneous_tube_95_coverage"),
         ("Endpoint XY RMSE [m]", "95% whole-path coverage")),
        ("distribution_match", ("path_energy_distance_m", "wasserstein1_path_m"),
         ("Path energy discrepancy [m]", "Path Wasserstein-1 [m]")),
    ):
        fig, axes = plt.subplots(2, len(dimensions), figsize=(4 * len(dimensions), 6.2), squeeze=False)
        for col, latent in enumerate(dimensions):
            for row, field in enumerate(fields):
                axis = axes[row, col]
                for option, color in ((1, "tab:blue"), (2, "tab:orange")):
                    axis.plot(horizons, [average(metrics[option], latent, horizon, field) for horizon in horizons],
                              "o-", color=color, label=f"Option {option}")
                if "coverage" in field:
                    axis.axhline(0.95, ls="--", color="0.4", lw=1)
                    axis.set_ylim(0, 1.03)
                axis.set(title=f"Latent {latent}", xlabel="Horizon [steps]", ylabel=labels[row], xticks=horizons)
                axis.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, len(dimensions), figsize=(4 * len(dimensions), 6.2), squeeze=False)
    for col, latent in enumerate(dimensions):
        for option in (1, 2):
            history = json.loads((root / f"option-{option}/latent-{latent}/history.json").read_text())
            axes[0, col].plot([r["step"] for r in history], [r["validation_nll_per_component"] for r in history],
                              label=f"Option {option}")
            summary = json.loads((root / f"option-{option}/latent-{latent}/summary.json").read_text())
            selected = json.loads((root / f"option-{option}/latent-{latent}" / summary["final_model"]
                                   / "manifest.json").read_text())["generation"]
            selected_row = next((row for row in history if row["step"] == selected), None)
            if selected_row is not None:
                axes[0, col].plot(selected, selected_row["validation_nll_per_component"], "kx", ms=7)
            field = "train_nll_per_component" if option == 1 else "train_expected_nll_per_component"
            axes[1, col].plot([r["step"] for r in history], [r[field] for r in history],
                              label=f"Option {option}: {'joint' if option == 1 else 'expected conditional'}")
        axes[0, col].set(title=f"Latent {latent}", ylabel="Validation joint NLL / component")
        axes[1, col].set(ylabel="Training loss (different objectives)")
        for axis in axes[:, col]:
            axis.set_xlabel("Neural updates")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "loss_curves_full.png", dpi=150)
    maximum = max(row["step"] for row in history)
    if maximum > 1000:
        for axis in axes.flat:
            values = np.concatenate([
                np.asarray(line.get_ydata(), dtype=float)[np.asarray(line.get_xdata()) >= 1000]
                for line in axis.lines if np.any(np.asarray(line.get_xdata()) >= 1000)
            ])
            values = values[np.isfinite(values)]
            if len(values):
                low, high = values.min(), values.max()
                margin = max(float(high - low) * 0.08, 0.05)
                axis.set(xlim=(1000, maximum), ylim=(low-margin, high+margin))
        fig.suptitle("Training after initial 1,000 updates; full curves saved separately")
        fig.tight_layout()
    fig.savefig(output / "loss_curves.png", dpi=150)
    plt.close(fig)
    from .trajectory_sections import plot_trajectory_sections

    records = plot_trajectory_sections(root, dimensions, metrics)
    primary = next((record for record in records if record["speed_mps"] == 15
                    and record["signal"] == 1 and record["start_step"] == 40), records[0])
    shutil.copyfile(root / primary["figure"], output / "trajectory_comparison.png")


def compare_training_options(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    dimensions = plan["latents"]
    metrics = {option: json.loads((root / f"evaluation-option-{option}/metrics.json").read_text())
               for option in (1, 2)}
    checks = pairing_checks(root, dimensions, metrics)
    fields = ("terminal_position_rmse_m", "simultaneous_tube_95_coverage", "path_energy_distance_m",
              "wasserstein1_path_m", "terminal_radius_95_m", "simultaneous_tube_radius_m",
              "physical_violation_fraction")
    rows = [{"option": option, "latent": latent, "horizon": horizon,
             **{field: average(metrics[option], latent, horizon, field) for field in fields},
             "endpoint_rmse_by_split": {split: average(metrics[option], latent, horizon,
                                                        "terminal_position_rmse_m", split)
                                         for split in ("train", "validation", "test")}}
            for option in (1, 2) for latent in dimensions for horizon in (1, 5, 10, 15)]
    write_json(root / "comparison.json", {"pairing_checks": checks, "rows": rows})
    plots(root, dimensions, metrics)
    counts = plan.get("split_counts", {"train": 27, "validation": 6, "test": 16})
    lines = ["# Vehicle BNN: option 1 + refit versus option 2", "",
             f"Dataset: {counts['train']} train / {counts['validation']} validation / "
             f"{counts['test']} unseen test physics; {plan['steps']:,} neural "
             f"updates per model; training seed {plan['seed']}, evaluation seed {plan['evaluation_seed']}. "
             "Same initial neural weights, normalization, architecture, actions, physical samples, and "
             "evaluation base-noise seed. One training seed per option/latent; not a multi-seed significance test.", "",
             "**Option 1:** unchanged subset-joint-gradient training, validation-NLL selection, then one "
             "train-only refit with 64 transitions per training vehicle. **Option 2:** full-instance offline "
             "Gaussian E-steps, expected-likelihood neural M-steps, population-moment updates/canonical "
             "whitening every 200 updates, and validation-rollout-energy selection. Equal neural update "
             "counts are not equal compute: option 2 additionally processes all training transitions in E-steps.", "",
             "## Results at 15 steps (1.5 seconds)", "",
             "| Latent | Option | Test endpoint RMSE [m] ↓ | 95% path coverage | Energy [m] ↓ | W1 [m] ↓ | 95% path radius [m] | Train wall time [s] |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if plan.get("dataset"):
        lines[4:4] = [f"[Dataset generation and provenance]({Path(plan['dataset']) / 'REPORT.md'}). "
                      f"Fingerprint: `{plan['dataset_fingerprint']}`.", ""]
    for latent in dimensions:
        for option in (1, 2):
            row = next(r for r in rows if (r["latent"], r["option"], r["horizon"]) == (latent, option, 15))
            wall = plan["timings"].get(f"option-{option}-latent-{latent}", 0)
            lines.append(f"| {latent} | {option} | {row['terminal_position_rmse_m']:.4f} | "
                         f"{row['simultaneous_tube_95_coverage']:.1%} | {row['path_energy_distance_m']:.4f} | "
                         f"{row['wasserstein1_path_m']:.4f} | {row['simultaneous_tube_radius_m']:.4f} | {wall:.1f} |")
    wins = {field: sum(average(metrics[2], latent, 15, field) < average(metrics[1], latent, 15, field)
                       for latent in dimensions) for field in ("terminal_position_rmse_m", "wasserstein1_path_m")}
    lines += ["", f"Option 2 has lower 15-step endpoint RMSE for **{wins['terminal_position_rmse_m']}/{len(dimensions)}** "
              f"latent sizes and lower W1 for **{wins['wasserstein1_path_m']}/{len(dimensions)}**. "
              "Coverage must be interpreted together with width and distance: 100% coverage can simply mean "
              "an overly broad distribution; neither method is certified for BRO chance constraints.", "",
              "![Error and coverage at 1/5/10/15 steps](plots/error_coverage.png)", "",
              "![Energy and Wasserstein at 1/5/10/15 steps](plots/distribution_match.png)", "",
              "## Seen and unseen physics", "",
              "Train physics are in-sample diagnostics; validation physics selected checkpoints; test physics "
              "were not used in fitting or selection. All use fresh evaluation controls.", "",
              "| Latent | Option | Train-condition RMSE [m] | Validation RMSE [m] | Test RMSE [m] |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["horizon"] == 15:
            values = row["endpoint_rmse_by_split"]
            lines.append(f"| {row['latent']} | {row['option']} | {values['train']:.4f} | "
                         f"{values['validation']:.4f} | {values['test']:.4f} |")
    lines += ["",
              "## Training and interpretation", "",
              "![Training losses and comparable joint validation NLL](plots/loss_curves.png)", "",
              "The two training-loss curves are different objectives (subset joint versus expected conditional "
              "NLL), so their absolute levels should not be compared. Validation curves score the same joint "
              "likelihood, with the stable QR calculation in option 2. Option 1's curves precede its final refit. "
              "Black crosses mark the selected neural checkpoints. "
              "Option 2's saved N(0,I) prior is a canonical coordinate choice: the learned population mean and "
              "covariance have been absorbed into the neural mean/basis. Each sampled function remains fixed "
              "throughout every rollout; observation noise is not injected into function samples.", "",
              "The final evaluation uses 24 matched fresh-control windows, 256 learned functions and 256 "
              "physical ST references. Endpoint RMSE measures the ensemble mean against individual test vehicles; "
              "it includes genuine between-vehicle variation. Energy and W1 compare the trajectory distributions "
              "using RMS XY path separation. The 95% coverage event covers the complete predicted path.", "",
              "Pairing checks passed: initial model fingerprints and all evaluation controls, truths, and "
              "physical-reference samples match exactly. This compares two complete procedures, not an ablation "
              "isolating the effect of QR, EM, whitening, or checkpoint selection individually.", "",
              "[Option 1 detailed report and state plots](evaluation-option-1/REPORT.md) · "
              "[Option 2 detailed report and state plots](evaluation-option-2/REPORT.md) · "
              "[All metrics and pairing checks](comparison.json) · [Resolved configurations](configs/) · "
              "[Training logs](logs/) · [Full loss curves](plots/loss_curves_full.png) · "
              "[Option 1 models](option-1/) · [Option 2 models](option-2/).", ""]
    violations = ["Sampled paths outside the collection envelope at 15 steps (option 1 / option 2): "]
    violations.append("; ".join(
        f"latent {latent}: {average(metrics[1], latent, 15, 'physical_violation_fraction'):.1%} / "
        f"{average(metrics[2], latent, 15, 'physical_violation_fraction'):.1%}" for latent in dimensions
    ) + ". These are steering/speed/yaw-rate/slip envelope checks, not collisions or BRO safety events.")
    lines[-2:-2] = ["".join(violations), ""]
    best = {option: min(dimensions, key=lambda latent: average(metrics[option], latent, 15, "wasserstein1_path_m"))
            for option in (1, 2)}
    leader = min((1, 2), key=lambda option: average(metrics[option], best[option], 15, "wasserstein1_path_m"))
    if all(value == 0 for value in wins.values()):
        verdict = "option 1 + refit is the stronger baseline at every tested latent size."
    elif all(value == len(dimensions) for value in wins.values()):
        verdict = "option 2 improves endpoint error and W1 at every tested latent size."
    else:
        verdict = "the methods have mixed results across latent sizes; inspect the per-model tradeoffs."
    conclusion = (
        f"**Practical conclusion for this budget:** {verdict} "
        f"The lowest observed W1 comes from option {leader}, latent {best[leader]} "
        f"(RMSE {average(metrics[leader], best[leader], 15, 'terminal_position_rmse_m'):.4f} m, "
        f"W1 {average(metrics[leader], best[leader], 15, 'wasserstein1_path_m'):.4f} m, "
        f"coverage {average(metrics[leader], best[leader], 15, 'simultaneous_tube_95_coverage'):.1%}). "
        f"Option 2's best observed distribution match is latent {best[2]}. "
        "Numerical correctness does not guarantee a better optimum: EM scheduling, optimizer "
        "restarts, and checkpoint selection remain coupled differences requiring validation-only "
        "ablations. Do not assume more updates or a flow will fix this."
    )
    index = lines.index("![Error and coverage at 1/5/10/15 steps](plots/error_coverage.png)")
    lines[index:index] = [conclusion, ""]
    section_index = lines.index("## Training and interpretation")
    from .trajectory_sections import sampled_trajectory_markdown

    extra_sections = sampled_trajectory_markdown(root)
    lines[section_index:section_index] = extra_sections
    lines[-2:-2] = ["Training wall times include process startup, preparation, compilation, validation, "
                     "and refit/EM overhead; they are not kernel-only timings.", ""]
    if (root / "performance-audit/README.md").is_file():
        lines[-2:-2] = ["Option 2 uses an algebraically equivalent pairwise covariance transform "
                       "to avoid quartic contraction cost at large latent dimensions. "
                       "[Performance benchmark and preserved partial run](performance-audit/README.md).", ""]
    if (root / "inference-audit/README.md").is_file():
        lines[-2:-2] = ["Unconditioned prediction and function sampling use the stored Gaussian "
                       "mean/factor directly, avoiding a lossy float32 covariance/precision round trip. "
                       "The combined latent-size report uses freshly reevaluated frozen baseline models. "
                       "[Inference audit](inference-audit/README.md).", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return root / "REPORT.md"
