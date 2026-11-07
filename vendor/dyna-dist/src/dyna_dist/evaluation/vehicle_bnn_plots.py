"""Static vehicle function-cover figures and a compact evaluation report."""

from pathlib import Path

import numpy as np


def _finite_paths(paths):
    paths = np.asarray(paths)
    return paths[np.isfinite(paths).all(axis=(1, 2))]


def _q(paths, component):
    return np.quantile(paths[:, :, component], [0.025, 0.5, 0.975], axis=0)


def _plot_xy_cover(root, speed, data, window, config, *, zoom=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .trajectory_sections import _draw_paths, TRUTH_COLOR, SAMPLE_COLOR, MEAN_COLOR

    truth = data["truth"]
    real = _finite_paths(data["real_model_paths"])
    latents = config.latent_dimensions
    fig, axes = plt.subplots(1, len(latents) + 1, figsize=(4.2 * (len(latents) + 1), 4.2))
    all_paths = [truth, real]
    if not zoom:
        all_paths += [_finite_paths(data[f"latent_{latent}_paths"]) for latent in latents]
    limits = np.concatenate([
        np.quantile(paths[..., :2].reshape(-1, 2), [0.01, 0.99], axis=0)
        for paths in all_paths if len(paths)
    ])
    low, high = limits.min(axis=0), limits.max(axis=0)
    span = np.maximum(high - low, 1.0)
    low, high = low - 0.05 * span, high + 0.05 * span
    final = window["metrics"][-1]
    for axis, latent in zip(axes[:-1], latents, strict=True):
        paths = _finite_paths(data[f"latent_{latent}_paths"])
        _draw_paths(axis, paths[..., :2], truth[..., :2])
        cover = final["bnn_cover"][str(latent)]
        score = cover.get("simultaneous_tube_95_coverage")
        axis.set_title(f"BNN latent {latent}\nGT whole-path tube cover: "
                       f"{score:.0%}" if score is not None else f"BNN latent {latent}\nunavailable")
    axis = axes[-1]
    _draw_paths(axis, real[..., :2], truth[..., :2])
    score = final["real_model_cover"].get("simultaneous_tube_95_coverage")
    axis.set_title(f"Sampled ST models\nGT whole-path tube cover: {score:.0%}")
    for axis in axes:
        axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]),
                 xlabel="x [m]", ylabel="y [m]")
        axis.set_aspect("auto" if zoom else "equal", adjustable="box")
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Nominal speed {speed:g} m/s; window starts at {truth[0, 0, 3]:.1f} m/s; "
        f"{len(truth)} held-out truths in orange; all saved samples in blue; mean in magenta"
        + ("; true-physics-scale x/y (independent axis scales)" if zoom
           else "; shared distribution scale")
    )
    fig.legend(handles=[Line2D([], [], color=TRUTH_COLOR, ls="--", label="Held-out test GT"),
                        Line2D([], [], color=SAMPLE_COLOR, label="Sampled functions"),
                        Line2D([], [], color=MEAN_COLOR, lw=2, label="Prediction mean")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    suffix = "_gt_zoom" if zoom else ""
    fig.savefig(root / "plots" / f"trajectory_cover_speed-{speed:g}{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_state_time(root, speed, data, config, timestep):
    import matplotlib.pyplot as plt

    truth = data["truth"]
    real = _finite_paths(data["real_model_paths"])
    if not len(real):
        return
    times = np.arange(truth.shape[1]) * timestep
    components = ((0, "x [m]"), (1, "y [m]"), (2, "steering [rad]"),
                  (3, "speed [m/s]"), (4, "heading [rad]"),
                  (5, "yaw rate [rad/s]"), (6, "slip [rad]"))
    for latent in config.latent_dimensions:
        learned = _finite_paths(data[f"latent_{latent}_paths"])
        if not len(learned):
            continue
        fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
        for axis, (component, label) in zip(axes.flat, components):
            for path in truth:
                axis.plot(times, path[:, component], color="0.5", alpha=0.25, lw=0.75)
            axis.plot(times, truth[0, :, component], color="black", lw=1,
                      label="held-out GT (one highlighted)")
            for paths, color, name in (
                (learned, "tab:blue", f"BNN latent {latent}"),
                (real, "tab:orange", "sampled ST"),
            ):
                low, center, high = _q(paths, component)
                axis.fill_between(times, low, high, color=color, alpha=0.14)
                axis.plot(times, center, color=color, lw=1.4, label=f"{name} median / 95%")
            axis.set(ylabel=label, xlabel="time [s]")
            axis.tick_params(labelbottom=True)
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        axes[-1, -1].set_visible(False)
        fig.suptitle(f"Fresh controls: nominal {speed:g} m/s, window starts at "
                     f"{truth[0, 0, 3]:.1f} m/s; truth vs sampled functions")
        fig.tight_layout()
        fig.savefig(
            root / "plots" / f"state_time_latent-{latent}_speed-{speed:g}.png", dpi=150
        )
        plt.close(fig)


def _rollout_mean(windows, horizon, key, latent=None):
    values = []
    for window in windows:
        row = next(item for item in window["metrics"] if item["horizon"] == horizon)
        cover = row["real_model_cover"] if latent is None else row["bnn_cover"][str(latent)]
        if cover.get(key) is not None:
            values.append(cover[key])
    return float(np.mean(values)) if values else None


def _plot_summary(root, config, windows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = (
        ("simultaneous_tube_95_coverage", "Held-out GT whole-path tube coverage", "fraction"),
        ("pointwise_xy_95_coverage", "Held-out GT pointwise XY coverage", "fraction"),
        ("mean_position_rmse_m", "Ensemble-mean position RMSE", "m"),
        ("terminal_radius_95_m", "Terminal sample 95% radius", "m"),
    )
    for axis, (field, title, unit) in zip(axes.flat, fields, strict=True):
        for latent in config.latent_dimensions:
            values = [_rollout_mean(windows, h, field, latent) for h in config.report_horizons]
            axis.plot(config.report_horizons, values, "o-", label=f"BNN {latent}")
        real = [_rollout_mean(windows, h, field) for h in config.report_horizons]
        axis.plot(config.report_horizons, real, "ko--", label="sampled ST")
        axis.set(title=title, xlabel="horizon [0.1 s steps]", ylabel=unit)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "plots" / "cover_vs_horizon.png", dpi=150)
    plt.close(fig)


def _plot_parameter_space(root, metrics):
    import matplotlib.pyplot as plt

    with np.load(root / "sampled_functions.npz", allow_pickle=False) as shard:
        reference = shard["real_parameter_values"]
    members = {}
    for row in metrics["one_step_rows"]:
        members[row["instance_id"]] = row
    keys = metrics["parameter_keys"]
    labels = ("friction μ", "front Csf multiplier", "rear Csr multiplier", "mass multiplier")
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2))
    palette = {"train": "tab:blue", "validation": "tab:green", "test": "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (x, y) in zip(axes.flat, pairs, strict=True):
        axis.scatter(reference[:, x], reference[:, y], s=7, color="0.75", alpha=0.35,
                     label="sampled ST reference")
        for split, color in palette.items():
            values = np.asarray([
                [row["parameters"][key] for key in keys]
                for row in members.values() if row["split"] == split
            ])
            axis.scatter(values[:, x], values[:, y], s=28, color=color,
                         label=split, edgecolor="white", linewidth=0.25)
        axis.set(xlabel=labels[x], ylabel=labels[y])
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Physics combinations: existing split and fresh sampled ST reference")
    fig.tight_layout()
    fig.savefig(root / "plots" / "parameter_space.png", dpi=150)
    plt.close(fig)


def plot_vehicle_bnn(root, config, metrics):
    import matplotlib
    import shutil

    matplotlib.use("Agg")
    root = Path(root)
    (root / "plots").mkdir(exist_ok=True)
    for latent in config.latent_dimensions:
        source = Path(config.training_run) / f"latent-{latent}" / "loss_curve.png"
        if source.is_file():
            shutil.copyfile(source, root / "plots" / f"training_loss_latent-{latent}.png")
    _plot_parameter_space(root, metrics)
    signal = config.rollout_signals[0]
    for speed in config.rollout_speeds:
        probe_id = f"speed-{speed:g}-signal-{signal}-start-{max(config.rollout_starts)}"
        window = next(w for w in metrics["rollout_windows"] if w["probe_id"] == probe_id)
        with np.load(root / "rollouts" / f"{probe_id}.npz", allow_pickle=False) as shard:
            data = {name: shard[name] for name in shard.files}
        _plot_xy_cover(root, speed, data, window, config)
        _plot_xy_cover(root, speed, data, window, config, zoom=True)
        _plot_state_time(root, speed, data, config, metrics["timestep"])
    _plot_summary(root, config, metrics["rollout_windows"])
    from .vehicle_bnn_summary import plot_compact_summary

    plot_compact_summary(root, config, metrics)
    if "mppi" in metrics:
        from .vehicle_mppi_plots import plot_mppi

        plot_mppi(root, config, metrics)


def write_vehicle_bnn_report(root, config, metrics):
    from .vehicle_bnn_summary import write_compact_report

    write_compact_report(root, config, metrics)
