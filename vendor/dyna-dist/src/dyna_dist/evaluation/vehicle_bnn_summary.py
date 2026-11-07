"""Compact report of endpoint accuracy and sampled trajectory distributions."""

from pathlib import Path

import numpy as np


def _average(metrics, horizon, key, latent=None, split=None):
    values = []
    for window in metrics["rollout_windows"]:
        row = next(item for item in window["metrics"] if item["horizon"] == horizon)
        cover = row["real_model_cover"] if latent is None else row["bnn_cover"][str(latent)]
        if split is not None:
            cover = cover["split_coverage"][split]
        if cover.get(key) is not None:
            values.append(cover[key])
    return np.mean(values, axis=0) if values else None


def _number(value, *, percent=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.1%}"
    return f"{value:.4f}" if abs(value) >= 0.001 else f"{value:.2e}"


def plot_compact_summary(root, config, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    (root / "plots").mkdir(exist_ok=True)
    horizons = config.report_horizons
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fields = ((None, "Position RMSE [m]"), (2, "Steering RMSE [rad]"),
              (3, "Speed RMSE [m/s]"), (4, "Heading RMSE [rad]"),
              (5, "Yaw-rate RMSE [rad/s]"), (6, "Slip RMSE [rad]"))
    for axis, (component, title) in zip(axes.flat, fields, strict=True):
        for latent in config.latent_dimensions:
            if component is None:
                values = [_average(metrics, h, "terminal_position_rmse_m", latent) for h in horizons]
            else:
                values = [_average(metrics, h, "terminal_state_rmse", latent)[component]
                          for h in horizons]
            axis.plot(horizons, values, "o-", label=f"latent {latent}")
        axis.set(title=title, xlabel="prediction step", xticks=horizons, yscale="log")
        axis.grid(alpha=0.2, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Error at the endpoint x_H: unseen test physics, fresh controls")
    fig.tight_layout()
    fig.savefig(root / "plots" / "short_horizon_errors.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fields = (("path_energy_distance_m", "XY path energy discrepancy [m]"),
              ("wasserstein1_path_m", "Exact XY path Wasserstein-1 [m]"),
              ("simultaneous_tube_95_coverage", "95% whole-path tube coverage"),
              ("terminal_radius_95_m", "Terminal 95% radius [m]"))
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        for latent in config.latent_dimensions:
            values = [_average(metrics, h, field, latent) for h in horizons]
            axis.plot(horizons, values, "o-", label=f"latent {latent}")
        axis.plot(horizons, [_average(metrics, h, field) for h in horizons],
                  "ko--", label="ST reference")
        axis.set(title=title, xlabel="prediction steps", xticks=horizons)
        if "coverage" in field:
            axis.axhline(0.95, color="0.6", ls=":", label="nominal 95%")
            axis.set_ylim(0, 1.05)
        else:
            axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed-function trajectory distribution: shared states and controls")
    fig.tight_layout()
    fig.savefig(root / "plots" / "short_horizon_distributions.png", dpi=150)
    plt.close(fig)


def write_compact_report(root, config, metrics):
    root = Path(root)
    windows = metrics["rollout_windows"]
    maximum = config.horizon
    best = min(config.latent_dimensions,
               key=lambda latent: _average(metrics, maximum, "wasserstein1_path_m", latent))
    best_error = _average(metrics, maximum, "terminal_position_rmse_m", best)
    best_cover = _average(metrics, maximum, "simultaneous_tube_95_coverage", best)
    lines = [
        "# Vehicle BNN: short-horizon accuracy and distribution match", "",
        f"Latents {', '.join(map(str, config.latent_dimensions))}; {len(windows)} fresh-control "
        f"cases at {', '.join(f'{speed:g}' for speed in config.rollout_speeds)} m/s nominal speeds; "
        f"{config.model_particles} functions/model and {config.real_models} ST reference functions. "
        f"One step is {metrics['timestep']:g} s. Each function stays fixed throughout a rollout, "
        "and models receive identical initial states and controls. No conditioning or retraining "
        "is performed; observation noise is excluded from function samples.", "",
        f"**Latent {best} has the best distribution match at {maximum} steps.** "
        f"Its endpoint position RMSE is {_number(best_error)} m; nominal 95% whole-path "
        f"coverage is {_number(best_cover, percent=True)}. "
        "High coverage must be read alongside distribution distance and width.", "",
        "## Prediction error", "",
        f"Position RMSE compares the ensemble mean at **x_H** with each of the "
        f"{len(windows[0]['test_instance_ids'])} unseen test-physics trajectories, "
        "then averages cases equally. These are endpoint "
        "errors, not errors averaged over x_1…x_H.", "",
        "| Latent | " + " | ".join(f"{h}-step RMSE [m]" for h in config.report_horizons) + " |",
        "| ---: | " + " | ".join("---:" for _ in config.report_horizons) + " |",
    ]
    for latent in config.latent_dimensions:
        values = [_number(_average(metrics, h, "terminal_position_rmse_m", latent))
                  for h in config.report_horizons]
        lines.append(f"| {latent} | " + " | ".join(values) + " |")
    lines += [
        "", f"At {maximum} steps, new controls on seen versus unseen physics give:", "",
        "| Latent | Seen/train RMSE [m] | Validation RMSE [m] | Test RMSE [m] |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for latent in config.latent_dimensions:
        values = [_number(_average(metrics, maximum, "terminal_position_rmse_m", latent, split))
                  for split in ("train", "validation", "test")]
        lines.append(f"| {latent} | " + " | ".join(values) + " |")
    lines += [
        "", "![Endpoint error in physical state components](plots/short_horizon_errors.png)", "",
        "## Coverage and distribution match", "",
        "Coverage is the fraction of held-out **entire XY paths** inside a tube whose "
        "radius is the 95th percentile of sampled paths' maximum distance from the ensemble "
        "mean. Radius is also reported so excessive spread cannot masquerade as accuracy.", "",
        "Energy discrepancy and exact empirical **Wasserstein-1** compare BNN versus ST "
        "path samples. Both use RMS XY separation over future states x_1…x_H as distance "
        "in meters; lower is better. Energy is twice the mean cross-distance minus the two "
        "mean within-ensemble distances. W1 uses optimal transport, not a Gaussian fit "
        "or a sliced approximation. **ST baseline** compares two independently generated "
        "physical ensembles with matching sample counts, showing finite-sample discrepancy; "
        "its coverage still uses the held-out test trajectories.", "",
        "| Steps | Model | 95% path coverage | Energy [m] ↓ | W1 [m] ↓ | Terminal 95% radius [m] |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in config.report_horizons:
        for label, latent in [(f"BNN {value}", value) for value in config.latent_dimensions] + [
            ("ST baseline", None)
        ]:
            values = [_average(metrics, horizon, field, latent) for field in (
                "simultaneous_tube_95_coverage", "path_energy_distance_m",
                "wasserstein1_path_m", "terminal_radius_95_m",
            )]
            lines.append(f"| {horizon} | {label} | " + " | ".join(
                _number(value, percent=index == 0) for index, value in enumerate(values)
            ) + " |")
    lines += ["", "![Coverage, energy, Wasserstein, and spread](plots/short_horizon_distributions.png)", ""]
    speed = config.rollout_speeds[len(config.rollout_speeds) // 2]
    example = f"plots/trajectory_cover_speed-{speed:g}_gt_zoom.png"
    if (root / example).is_file():
        lines += [f"Representative turning window at nominal {speed:g} m/s; axes use the "
                  "physical-trajectory scale, so divergent BNN paths are clipped visually.", "",
                  f"![Ground truth versus sampled trajectories]({example})", ""]
    lines += [
        "## Interpretation", "",
        "Use error and distribution distances to judge model quality; coverage alone rewards "
        "overly broad models. The physical target here is uniform sampling over the declared "
        "friction, front/rear stiffness, and mass ranges, rather than an identified deployment "
        "posterior. These measurements do not certify BRO chance constraints.", "",
        "[All plots, including state traces and training curves](plots/) · "
        "[Full metrics, per-case and per-split coverage](metrics.json) · "
        "[Resolved configuration](config.yaml). "
        "Existing MPPI diagnostics remain in [mppi/](mppi/).", "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
