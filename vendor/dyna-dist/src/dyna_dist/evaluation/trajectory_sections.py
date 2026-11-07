"""Multiple matched short trajectory sections rendered from saved predictions."""

import json
from pathlib import Path

import numpy as np

from dyna_dist.io.common import write_json


TRUTH_COLOR = "#D55E00"
SAMPLE_COLOR = "#0072B2"
MEAN_COLOR = "#A100A1"


def lateral_offsets(paths, reference):
    """Signed displacement from the test-mean path, for display only."""
    tangent = np.gradient(reference[:, :2], axis=0)
    length = np.linalg.norm(tangent, axis=-1)
    tangent = np.divide(tangent, length[:, None], out=np.zeros_like(tangent),
                        where=length[:, None] > 1e-12)
    tangent[length <= 1e-12] = (1, 0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return np.einsum("pti,ti->pt", paths[..., :2] - reference[None, :, :2], normal)


def _draw_paths(axis, samples, truth):
    """Consistent semantic colors, with every finite saved function displayed."""
    from matplotlib.collections import LineCollection

    axis.add_collection(LineCollection(samples, colors=SAMPLE_COLOR, linewidths=0.65,
                                       alpha=0.18, zorder=1))
    axis.add_collection(LineCollection(truth, colors=TRUTH_COLOR, linewidths=1.1,
                                       linestyles="dashed", alpha=0.85, zorder=3))
    axis.scatter(truth[:, -1, 0], truth[:, -1, 1], facecolors="none", edgecolors=TRUTH_COLOR,
                 linewidths=0.8, s=20, zorder=5)
    if len(samples):
        center = samples.mean(axis=0)
        axis.plot(center[:, 0], center[:, 1], color=MEAN_COLOR, lw=2.1, zorder=4)
        axis.scatter(samples[:, -1, 0], samples[:, -1, 1], color=SAMPLE_COLOR,
                     alpha=0.28, s=5, zorder=2)
        axis.scatter(center[-1, 0], center[-1, 1], marker="D", color=MEAN_COLOR, s=22, zorder=6)
    axis.plot(truth[0, 0, 0], truth[0, 0, 1], "kx", ms=4, zorder=7)


def sampled_trajectory_markdown(root, per_speed=2):
    """Embed several diverse cached cases instead of a single representative path."""
    root = Path(root)
    manifest = root / "trajectory_sections.json"
    if not manifest.is_file():
        return []
    records = json.loads(manifest.read_text())["sections"]
    selected = []
    for speed in sorted({record["speed_mps"] for record in records}):
        group = sorted([record for record in records if record["speed_mps"] == speed],
                       key=lambda record: (record["start_step"], record["signal"]))
        indices = np.linspace(0, len(group) - 1, min(per_speed, len(group))).astype(int)
        if per_speed == 1:
            indices = [len(group) - 1]
        selected.extend(group[index] for index in indices)
    if not selected:
        return []
    lines = ["## Sampled trajectory comparisons", "",
             "**Orange dashed:** held-out test GT (open-circle endpoints). **Blue:** all saved "
             "function samples, not just the former 64. **Magenta:** prediction mean (diamond endpoint). "
             "Every function remains fixed across the prediction horizon. Shared panel limits enable "
             "direct comparison of the two options and all latent sizes; XY axes use independent scaling.", "",
             f"{len(selected)} representative sections below; "
             f"[all {len(records)} sections with lateral-spread details](TRAJECTORY_SECTIONS.md).", ""]
    for record in selected:
        lines += [f"### {record['speed_mps']:g} m/s · signal {record['signal']} · "
                  f"start {record['start_time_s']:g} s", "",
                  f"![Sampled trajectories: {record['probe_id']}]({record['figure']})", ""]
    detail = selected[min(len(selected) // 2 + 1, len(selected) - 1)]
    if detail.get("lateral_figure"):
        lines += ["### Spread detail", "",
                  "Small XY differences can be hidden by the distance travelled. This view projects "
                  "each path onto the local normal of the test-mean path at each prediction step. "
                  "The GT reference is used only to display displacement, never as model input.", "",
                  f"![Lateral spread: {detail['probe_id']}]({detail['lateral_figure']})", ""]
    return lines


def split_audit(root, evaluation_config=None):
    from dyna_dist.config import VehicleBNNEvaluationConfig, load_config

    config = load_config(VehicleBNNEvaluationConfig,
                         evaluation_config or root / "evaluation-option-1/config.yaml")
    manifest = json.loads((Path(config.dataset) / "manifest.json").read_text())
    keys = ("friction", "front_cornering_scale", "rear_cornering_scale", "mass_ratio")
    members = {split: [member for member in manifest["instances"] if member["split"] == split]
               for split in ("train", "validation", "test")}
    values = {split: {tuple(member[key] for key in keys) for member in items}
              for split, items in members.items()}
    overlap = {f"{a}/{b}": len(values[a] & values[b])
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return {"counts": {split: len(items) for split, items in members.items()},
            "exact_combination_overlap": overlap,
            "validation_combinations": [{"instance_id": item["instance_id"],
                                          **{key: item[key] for key in keys}}
                                         for item in members["validation"]]}


def plot_trajectory_sections(root, dimensions, metrics, *, source_roots=None, evaluation_config=None):
    """One paired 2-by-latent figure per section; no model/simulator execution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(root)
    output = root / "plots/trajectory_sections"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for window in metrics[1]["rollout_windows"]:
        data = {}
        for option in (1, 2):
            data[option] = {}
            loaded = {}
            for latent in dimensions:
                source = Path(source_roots[latent]) if source_roots else root
                if source not in loaded:
                    with np.load(source / f"evaluation-option-{option}/rollouts/{window['probe_id']}.npz",
                                 allow_pickle=False) as shard:
                        loaded[source] = {key: shard[key] for key in ("truth_test", "controls")}
                        loaded[source].update({f"latent_{dimension}_paths": shard[f"latent_{dimension}_paths"]
                                               for dimension in dimensions if f"latent_{dimension}_paths" in shard})
                values = loaded[source]
                if data[option]:
                    np.testing.assert_array_equal(data[option]["truth_test"], values["truth_test"])
                    np.testing.assert_array_equal(data[option]["controls"], values["controls"])
                else:
                    data[option].update({key: values[key] for key in ("truth_test", "controls")})
                data[option][f"latent_{latent}_paths"] = values[f"latent_{latent}_paths"]
        truth = data[1]["truth_test"]
        particle_count = len(data[1][f"latent_{dimensions[0]}_paths"])
        np.testing.assert_array_equal(truth, data[2]["truth_test"])
        np.testing.assert_array_equal(data[1]["controls"], data[2]["controls"])
        # A shared view includes GT and central 95% sampled XY positions from
        # every compared model; neither method gets its own convenient scale.
        extents = [truth[..., :2].reshape(-1, 2)]
        reference = truth.mean(axis=0)
        truth_lateral = lateral_offsets(truth, reference)
        lateral_extents = [truth_lateral.ravel()]
        finite_paths = {}
        for option in (1, 2):
            for latent in dimensions:
                paths = data[option][f"latent_{latent}_paths"]
                finite = paths[np.isfinite(paths).all(axis=(1, 2))]
                finite_paths[option, latent] = finite
                if len(finite):
                    extents.append(np.quantile(finite[..., :2], (0.025, 0.975), axis=0).reshape(-1, 2))
                    lateral_extents.append(np.quantile(lateral_offsets(finite, reference),
                                                       (0.025, 0.975), axis=0).ravel())
        extent = np.concatenate(extents)
        low, high = extent.min(axis=0), extent.max(axis=0)
        padding = np.maximum(high - low, 0.3) * 0.08
        fig, axes = plt.subplots(2, len(dimensions), figsize=(4 * len(dimensions), 6.5), squeeze=False)
        for row, option in enumerate((1, 2)):
            for column, latent in enumerate(dimensions):
                axis = axes[row, column]
                paths = finite_paths[option, latent]
                _draw_paths(axis, paths[..., :2], truth[..., :2])
                axis.set(title=f"Option {option}, latent {latent}", xlabel="x [m]", ylabel="y [m]",
                         xlim=(low[0]-padding[0], high[0]+padding[0]),
                         ylim=(low[1]-padding[1], high[1]+padding[1]))
                axis.grid(alpha=0.2)
        timestep = metrics[1]["timestep"]
        start_time = window["start_step"] * timestep
        duration = (truth.shape[1] - 1) * timestep
        fig.suptitle(f"Nominal {window['speed_mps']:g} m/s · signal {window['signal']} · "
                     f"section {start_time:g}–{start_time+duration:g} s\n"
                     f"GT: {len(truth)} unseen TEST physics · all {particle_count} function samples/model · "
                     "shared limits; independent x/y scaling")
        handles = [Line2D([], [], color=TRUTH_COLOR, ls="--", lw=1.5, marker="o",
                          markerfacecolor="none", label="Held-out test GT"),
                   Line2D([], [], color=SAMPLE_COLOR, lw=1, label="Sampled functions"),
                   Line2D([], [], color=MEAN_COLOR, lw=2.1, marker="D", label="Prediction mean")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        relative = f"plots/trajectory_sections/{window['probe_id']}.png"
        fig.savefig(root / relative, dpi=150)
        plt.close(fig)
        lateral_limits = np.concatenate(lateral_extents)
        bottom, top = lateral_limits.min(), lateral_limits.max()
        lateral_padding = max(top - bottom, 0.01) * 0.08
        times = np.arange(truth.shape[1]) * timestep
        truth_detail = np.stack((np.broadcast_to(times, truth_lateral.shape), truth_lateral), axis=-1)
        fig, axes = plt.subplots(2, len(dimensions), figsize=(4 * len(dimensions), 6.5), squeeze=False)
        for row, option in enumerate((1, 2)):
            for column, latent in enumerate(dimensions):
                axis = axes[row, column]
                offsets = lateral_offsets(finite_paths[option, latent], reference)
                detail = np.stack((np.broadcast_to(times, offsets.shape), offsets), axis=-1)
                _draw_paths(axis, detail, truth_detail)
                axis.axhline(0, color="0.5", ls=":", lw=0.8, zorder=0)
                axis.set(title=f"Option {option}, latent {latent}", xlabel="Prediction time [s]",
                         ylabel="Lateral offset from GT mean [m]", xlim=(times[0], times[-1]),
                         ylim=(bottom-lateral_padding, top+lateral_padding))
                axis.grid(alpha=0.2)
        fig.suptitle(f"Lateral-spread detail · {window['speed_mps']:g} m/s · signal {window['signal']} · "
                     f"start {start_time:g} s\nAll {particle_count} function samples · "
                     "same displacement scale in every panel")
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        lateral_relative = f"plots/trajectory_sections/{window['probe_id']}_lateral.png"
        fig.savefig(root / lateral_relative, dpi=150)
        plt.close(fig)
        records.append({"probe_id": window["probe_id"], "speed_mps": window["speed_mps"],
                        "signal": window["signal"], "start_step": window["start_step"],
                        "start_time_s": start_time, "duration_s": duration,
                        "truth_split": "test", "truth_instances": len(truth),
                        "function_particles": particle_count, "figure": relative,
                        "lateral_figure": lateral_relative,
                        "displayed_function_samples": {
                            f"option-{option}-latent-{latent}": len(paths)
                            for (option, latent), paths in finite_paths.items()
                        }})
    audit = split_audit(root, evaluation_config)
    write_json(root / "trajectory_sections.json", {
        "split_audit": audit, "sections": records,
        "style": {"ground_truth": TRUTH_COLOR, "function_samples": SAMPLE_COLOR, "mean": MEAN_COLOR},
        "source_studies": {str(dimension): str(Path(source_roots[dimension]).resolve())
                           for dimension in dimensions} if source_roots else None,
    })
    lines = ["# Matched trajectory sections", "",
             f"Orange dashed trajectories and open-circle endpoints are the "
             f"**{audit['counts']['test']} held-out test physics combinations**; "
             "blue curves show **all saved function samples**, not just 64. The thick magenta curve "
             "and diamond endpoint show the prediction mean. "
             "Each function stays fixed over its complete prediction section. Both options and "
             "all latent sizes use identical initial states, actions, GT, and panel limits within a section. "
             "Panels use independent x/y scaling, not equal-distance map geometry. Limits include GT and "
             "central 95% sampled positions; extreme function samples may lie outside the view. "
             "Nonfinite paths are excluded from drawing. A second view per section shows signed lateral "
             "displacement relative to the test-mean path, exposing small spread/bias hidden by the XY "
             "distance scale. This GT reference is for visualization only, never model input.", "",
             "## Train versus validation", "",
             f"Verified split: {audit['counts']['train']} training, {audit['counts']['validation']} validation, "
             f"and {audit['counts']['test']} test physics combinations. Exact four-parameter overlaps: "
             + "; ".join(f"{key} = {value}" for key, value in audit["exact_combination_overlap"].items()) + ".", "",
             "The validation combinations are **not training inputs**: they do not fit the normalizer, "
             "neural gradients, local training posteriors, or population-prior refit. Their scores do select "
             "checkpoints, so validation is not an untouched final test. The individual parameter ranges "
             "overlap training: these are held-out combinations inside the same ranges (interpolation), "
             "not extrapolation to new parameter ranges. Original dataset action tapes are shared across "
             "physics by design; the sections below instead use the saved fresh evaluation tapes.", "",
             "| Validation instance | Friction | Front stiffness scale | Rear stiffness scale | Mass ratio |",
             "| :--- | ---: | ---: | ---: | ---: |"]
    for member in audit["validation_combinations"]:
        lines.append(f"| {member['instance_id']} | {member['friction']:.5f} | "
                     f"{member['front_cornering_scale']:.5f} | {member['rear_cornering_scale']:.5f} | "
                     f"{member['mass_ratio']:.5f} |")
    lines += ["", f"## All {len(records)} sections", "",
              "No training or prediction was rerun; these figures use the previously saved trajectories.", ""]
    for record in records:
        lines += [f"### {record['speed_mps']:g} m/s, signal {record['signal']}, "
                  f"start {record['start_time_s']:g} s", "",
                  f"![{record['probe_id']}]({record['figure']})", "",
                  f"![Lateral-spread detail: {record['probe_id']}]({record['lateral_figure']})", ""]
    (root / "TRAJECTORY_SECTIONS.md").write_text("\n".join(lines))
    return records
