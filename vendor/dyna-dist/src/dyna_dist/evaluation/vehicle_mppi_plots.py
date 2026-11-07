"""Static MPPI decision and distribution diagnostics embedded by the report."""

import numpy as np


def _mean(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def plot_mppi(root, config, metrics):
    import matplotlib.pyplot as plt

    mppi = metrics["mppi"]
    rows = mppi["rows"]
    palette = plt.get_cmap("tab10")
    latents = config.latent_dimensions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = ("spearman_rank", "top_10_percent_overlap", "selected_candidate_cost_gap",
            "weighted_proposal_cost_gap")
    titles = ("Candidate cost rank agreement", "Best-decile overlap",
              "Selected candidate: reference cost above best", "MPPI weighted proposal: reference cost gap")
    for axis, key, title in zip(axes.flat, keys, titles, strict=True):
        for i, latent in enumerate(latents):
            values = [row[key] for row in rows if row["latent"] == latent and row[key] is not None]
            axis.scatter(np.full(len(values), i), values, alpha=0.45, s=22, color=palette(i))
            axis.plot(i, np.mean(values), "k_", ms=16, mew=2)
        axis.set(title=title, xticks=range(len(latents)), xticklabels=latents,
                 xlabel="latent dimension")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "plots" / "mppi_decisions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(latents), figsize=(4 * len(latents), 4), squeeze=False)
    bins = np.linspace(0, 1, 11)
    for axis, latent in zip(axes.flat, latents, strict=True):
        estimated, actual = [], []
        for row in rows:
            if row["latent"] != latent:
                continue
            with np.load(root / "mppi" / f"{row['probe_id']}.npz", allow_pickle=False) as shard:
                estimated.append(shard[f"latent_{latent}_probabilities"])
                actual.append(shard["reference_probabilities"])
        estimated, actual = np.concatenate(estimated), np.concatenate(actual)
        for width_index, width in enumerate(config.mppi_half_widths):
            predicted, truth = estimated[:, width_index], actual[:, width_index]
            axis.scatter(predicted[::16], truth[::16], s=3, alpha=0.05)
            x, y = [], []
            for left, right in zip(bins[:-1], bins[1:], strict=True):
                selected = (predicted >= left) & (predicted < right if right < 1 else predicted <= 1)
                if selected.any():
                    x.append(predicted[selected].mean())
                    y.append(truth[selected].mean())
            axis.plot(x, y, "o-", ms=3, label=f"half-width {width:g} m")
        axis.plot([0, 1], [0, 1], "k--", lw=1)
        axis.set(title=f"BNN latent {latent}", xlabel="estimated corridor-exit probability",
                 ylabel="physical-reference exit probability", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "plots" / "mppi_probability_calibration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, latent in enumerate(latents):
        bench = [row for row in mppi["benchmark"]["rows"] if row["latent"] == latent]
        axes[0].plot([r["particles"] for r in bench], [r["steady_state_ms"] for r in bench],
                     "o-", color=palette(i), label=f"latent {latent}")
        width = []
        for horizon in config.report_horizons:
            subset = [next(row for row in window["metrics"] if row["horizon"] == horizon)
                      for window in metrics["rollout_windows"]]
            width.append(np.mean([
                row["bnn_cover"][str(latent)]["terminal_radius_95_m"]
                / max(row["real_model_cover"]["terminal_radius_95_m"], 1e-9)
                for row in subset
            ]))
        axes[1].plot(config.report_horizons, width, "o-", color=palette(i), label=f"latent {latent}")
    axes[0].set(title="Steady-state GPU prediction time", xlabel="fixed dynamics functions",
                ylabel="milliseconds / candidate batch")
    axes[0].legend(fontsize=8)
    axes[1].set(title="BNN / physical terminal 95% radius", xlabel="prediction steps", ylabel="spread ratio")
    axes[1].axhline(1, color="k", ls="--")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "plots" / "mppi_runtime_and_spread.png", dpi=150)
    plt.close(fig)

    for split in ("train", "test"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, name in enumerate(["physical"] + [f"latent_{value}" for value in latents]):
            with np.load(root / "mppi" / f"replan-{split}-{name}.npz", allow_pickle=False) as shard:
                states, reference, actions = shard["states"], shard["reference"], shard["actions"]
            color = "black" if name == "physical" else palette(i - 1)
            label = "physical-model MPPI" if name == "physical" else name.replace("_", " ")
            times = np.arange(len(states)) * metrics["timestep"]
            axes[0].plot(states[:, 0], states[:, 1], color=color, label=label)
            error = states[:, :2] - reference[:, :2]
            axes[1].plot(times, np.linalg.norm(error, axis=-1), color=color)
            axes[2].plot(times[:-1], actions[:, 0], color=color)
        axes[0].plot(reference[:, 0], reference[:, 1], "r--", label="target")
        axes[0].set(title=f"{split} physics: first-action execution", xlabel="x [m]", ylabel="y [m]")
        axes[0].legend(fontsize=7)
        axes[1].set(title="Executed position error", xlabel="time [s]", ylabel="m")
        axes[2].set(title="Executed steering rate", xlabel="time [s]", ylabel="rad/s")
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "plots" / f"mppi_replanning_{split}.png", dpi=150)
        plt.close(fig)


def mppi_report_lines(config, metrics):
    mppi = metrics["mppi"]
    rows = mppi["rows"]
    lines = [
        "## MPPI candidate decisions", "",
        f"The {len({row['probe_id'] for row in rows})} decision cases use the first fresh "
        "control tape at each speed and both configured start states. "
        f"Each case uses {config.mppi_candidates} common candidate control sequences and "
        f"{config.mppi_particles} fixed BNN functions. Perturbations use BRO's truncated "
        "Gaussian rule, normalized standard deviations, and physical action scales. "
        "Each coefficient stays fixed over time and across candidate controls.", "",
        "The standalone task follows a curved reference formed by the mean physical "
        "nominal path. The same reference and cost apply to all models: " + mppi["task"] + ".", "",
        "The cost gap of a selected sampled candidate is relative to the best physical-reference "
        "candidate. The weighted-proposal gap compares the physical cost of each model's "
        "importance-weighted control sequence with physical-model MPPI's weighted sequence; "
        "it can be negative. These are dynamics diagnostics, not BRO lap-performance scores.", "",
        "| Latent | Rank correlation ↑ | Best-decile overlap ↑ | Selected candidate cost gap ↓ | "
        "Weighted proposal cost gap ↓ |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for latent in config.latent_dimensions:
        selected = [row for row in rows if row["latent"] == latent]
        values = [_mean(selected, key) for key in ("spearman_rank", "top_10_percent_overlap",
                  "selected_candidate_cost_gap", "weighted_proposal_cost_gap")]
        lines.append(f"| {latent} | " + " | ".join(
            "unavailable" if value is None else f"{value:.4f}" for value in values
        ) + " |")
    lines += ["", "![Candidate decision quality](plots/mppi_decisions.png)", "",
              "## Probability of leaving a local corridor", "",
              "Each function counts once if any predicted state leaves the corridor. "
              "Non-finite paths count unsafe. Corridor widths are synthetic test conditions; "
              "this evaluation does not create a BRO map or terminal safe-set hull.", "",
              f"The physical reference uses {mppi['reference_particles']} models per candidate "
              f"(probability increments {mppi['probability_resolution']:.4f}). "
              "Its finite-sample estimate is a reference, not an exact probability. "
              "Curves pool candidate controls within cases; candidates are correlated.", "",
              "Table values average cases equally. Conditional false-safe rates average "
              "only cases with at least one predicted-safe candidate; raw counts are saved.", "",
              "| Latent | Corridor half-width [m] | Probability MAE ↓ | False-safe / all candidates | "
              "False-safe / predicted-safe candidates |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for latent in config.latent_dimensions:
        selected = [row for row in rows if row["latent"] == latent]
        for index, width in enumerate(config.mppi_half_widths):
            mae = np.mean([r["probability_mae_by_width"][index] for r in selected])
            all_rate = np.mean([r["false_safe_fraction_of_all_by_width"][index] for r in selected])
            conditional = [r["false_safe_fraction_of_predicted_safe_by_width"][index]
                           for r in selected if r["false_safe_fraction_of_predicted_safe_by_width"][index]
                           is not None]
            safe_rate = f"{np.mean(conditional):.1%}" if conditional else "no predicted-safe candidates"
            lines.append(f"| {latent} | {width:g} | {mae:.3f} | {all_rate:.1%} | {safe_rate} |")
    lines += ["", "![Local-corridor probability calibration](plots/mppi_probability_calibration.png)",
              "", "## First-action execution and replanning", "",
              f"The controller replans for {config.horizon} steps, executes the first action in "
              f"a fixed physical ST model, then observes and replans for {config.replan_steps} "
              f"real control steps. This small diagnostic uses {config.replan_candidates} "
              "candidates and one training-physics and one test-physics instance. "
              "It is a limited control sanity check rather than a full racing benchmark.", "",
              "| Physics | Predictor | Executed position RMSE [m] | Maximum cross-track error [m] |",
              "| :--- | :--- | ---: | ---: |"]
    for row in mppi["replanning"]:
        lines.append(f"| {row['split']} | {row['model']} | {row['position_rmse_m']:.4f} | "
                     f"{row['max_cross_track_m']:.4f} |")
    lines += ["", "![Replanning on seen physics](plots/mppi_replanning_train.png)", "",
              "![Replanning on unseen physics](plots/mppi_replanning_test.png)", "",
              "## Prediction throughput", "",
              "Device: " + ", ".join(mppi["benchmark"]["devices"]) + ". "
              "Timing uses three synchronized repetitions after the first call. "
              "This measures the rollout kernel only, excluding scoring, hulls, transfers and I/O. "
              "First-call latency is saved separately in the metrics.", "",
              "| Latent | Function samples | Candidate controls | Horizon | Steady-state prediction [ms] |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in mppi["benchmark"]["rows"]:
        lines.append(f"| {row['latent']} | {row['particles']} | {row['candidates']} | "
                     f"{row['horizon']} | {row['steady_state_ms']:.2f} |")
    lines += ["", "![Prediction time and relative spread](plots/mppi_runtime_and_spread.png)", ""]
    return lines
