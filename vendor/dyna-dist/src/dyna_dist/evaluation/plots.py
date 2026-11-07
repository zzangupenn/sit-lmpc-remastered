"""Optional offline Matplotlib diagnostics with tight layout."""

import numpy as np


def plot_training_history(history, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    steps = [row["step"] for row in history]
    for axis, field, title in zip(
        axes,
        ("train_nll_per_component", "validation_nll_per_component"),
        ("Training NLL", "Held-out validation NLL"),
        strict=True,
    ):
        axis.plot(steps, [row[field] for row in history], "o-", markersize=2)
        axis.set(xlabel="Optimizer updates", ylabel="Normalized NLL / component", title=title)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_evaluation(rows, plot_data, output, domain="st"):
    import matplotlib.pyplot as plt

    if domain in ("halfcheetah", "walker2d", "dm_walker"):
        return plot_halfcheetah(rows, plot_data, output, domain=domain)

    contexts = sorted({row["context_size"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for component, label in ((0, "x"), (1, "y"), (5, "yaw rate"), (6, "slip")):
        means = [
            np.mean(
                [
                    row["rmse"][component]
                    for row in rows
                    if row["context_size"] == context and row["rmse"] is not None
                ]
            )
            for context in contexts
        ]
        axes[0].plot(contexts, means, "o-", label=label)
    axes[0].set(xlabel="Observed context transitions", ylabel="One-step RMSE (component units)")
    axes[0].legend()
    if plot_data is not None:
        initial, _, truth, paths, _ = plot_data
        for path in paths[:40]:
            axes[1].plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.15)
        axes[1].plot(truth[:, 0], truth[:, 1], "k.-", label="Gym truth")
        axes[1].plot(initial[0], initial[1], "ro", label="Initial state")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].legend()
    axes[1].set(xlabel="x (m)", ylabel="y (m)")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)


def plot_halfcheetah(rows, plot_data, output, domain="halfcheetah"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.ravel()
    contexts = sorted({row["context_size"] for row in rows})
    components = ((0, "root x"), (2, "torso angle"), (9, "forward speed"))
    if contexts == [0]:
        labels = [label for _, label in components]
        values = [
            np.mean([row["rmse"][component] for row in rows if row["rmse"] is not None])
            for component, _ in components
        ]
        axes[0].bar(labels, values, color=("#0072B2", "#D55E00", "#009E73"))
        axes[0].tick_params(axis="x", labelrotation=15)
        axes[0].set(ylabel="One-step RMSE (component units)", title="Population mean")
    else:
        for component, label in components:
            values = [
                np.mean(
                    [
                        row["rmse"][component]
                        for row in rows
                        if row["context_size"] == c and row["rmse"] is not None
                    ]
                )
                for c in contexts
            ]
            axes[0].plot(contexts, values, "o-", label=label)
        axes[0].set(xlabel="Context transitions", ylabel="One-step RMSE (component units)")
        axes[0].legend()
    if plot_data is not None:
        _, _, truth, paths, _ = plot_data
        for axis, (index, label) in zip(
            axes[1:],
            (
                (0, "root x (m)"),
                (9, "forward speed (m/s)"),
                (2, "torso angle (rad)"),
                (3, "right thigh (rad)" if domain == "walker2d" else "back thigh (rad)"),
                (6, "left thigh (rad)" if domain == "walker2d" else "front thigh (rad)"),
            ),
            strict=True,
        ):
            steps = np.arange(1, len(truth) + 1)
            for sample in paths[:24]:
                axis.plot(steps, sample[:, index], color="tab:blue", alpha=0.1)
            low, high = np.quantile(paths[:, :, index], [0.025, 0.975], axis=0)
            axis.fill_between(
                steps, low, high, color="tab:blue", alpha=0.15, label="95% epistemic samples"
            )
            axis.plot(steps, paths[:, :, index].mean(0), color="tab:blue", label="Sample mean")
            axis.plot(steps, truth[:, index], "k.-", label="MuJoCo truth")
            axis.set(xlabel="Rollout step", ylabel=label)
        axes[1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=150)
    plt.close(fig)
