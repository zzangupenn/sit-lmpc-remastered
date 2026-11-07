"""Paired vehicle ST collection using vehicle_data_gen's smooth targets and PID law.

The source script is intentionally not imported: it executes configuration and
environment setup at import time. This adaptation uses explicit RNGs, complete
transition shards, and one shared action tape per speed/signal probe.
"""

from dataclasses import asdict, replace
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

from dyna_dist.adapters.gym_st import GymST
from dyna_dist.config import resolve_seed, write_config
from dyna_dist.io.common import fingerprint, new_directory, versions, write_json
from dyna_dist.io.dataset import DATASET_VERSION
from dyna_dist.models.profiles import CONTROL_FIELDS, STATE_FIELDS, vehicle_parameters


KNOBS = ("mu", "front", "rear", "mass")


def parameter_instances(config, rng):
    """One-at-a-time anchors plus space-filling, held-out parameter combinations."""
    bounds = np.array(
        [config.friction, config.front_cornering_scale,
         config.rear_cornering_scale, config.mass_ratio],
        dtype=float,
    )
    middle = np.mean(bounds, axis=1)
    result = [(middle.copy(), "train", "baseline")]
    for axis, name in enumerate(KNOBS):
        for end, label in enumerate(("low", "high")):
            values = middle.copy()
            values[axis] = bounds[axis, end]
            result.append((values, "train", f"{name}-{label}"))

    count = config.random_instances
    unit = np.column_stack(
        [(rng.permutation(count) + rng.random(count)) / count for _ in KNOBS]
    )
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    split_order = rng.permutation(count)
    n_train = count - 2 * max(1, round(0.15 * count))
    n_validation = max(1, round(0.15 * count))
    splits = np.full(count, "test", dtype=object)
    splits[split_order[:n_train]] = "train"
    splits[split_order[n_train:n_train + n_validation]] = "validation"
    result.extend((row, str(splits[i]), None) for i, row in enumerate(values))
    return result


def _smooth_steering_targets(steps, count, rng):
    """Sum random sinusoids as get_steers does in vehicle_data_gen."""
    x = np.linspace(0.0, 1.0, steps)
    signal = np.zeros(steps)
    for _ in range(count):
        amplitude = rng.random()
        frequency = rng.integers(1, count)
        phase = rng.uniform(0.0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * frequency * x + phase)
    span = np.ptp(signal)
    return np.zeros_like(signal) if span == 0 else 2 * (signal - signal.min()) / span - 1


def _pid(target_speed, target_steer, state, parameters):
    """The original vehicle_data_gen PID law, with physical control clipping below."""
    speed, steer = float(state[3]), float(state[2])
    steer_error = target_steer - steer
    steering_speed = np.sign(steer_error) * min(abs(steer_error) / 0.1, parameters["sv_max"])
    speed_error = target_speed - speed
    if speed > 0:
        gain = 10 * parameters["a_max"] / (
            parameters["v_max"] if speed_error > 0 else -parameters["v_min"]
        )
    else:
        gain = 2 * parameters["a_max"] / (
            parameters["v_max"] if speed_error > 0 else -parameters["v_min"]
        )
    return gain * speed_error, steering_speed


def _initial(speed):
    return np.array([0.0, 0.0, 0.0, speed, 0.0, 0.0, 0.0], dtype=np.float64)


def shared_probe(config, speed, rng, steps=None):
    """Record a nominal closed-loop action tape, then replay it open-loop everywhere."""
    parameters = vehicle_parameters(
        "fullscale", friction=sum(config.friction) / 2, mass_ratio=1.0,
        inertia_policy="fixed",
    )
    simulator = GymST(parameters, config.timestep, config.integrator_timestep)
    state = _initial(speed)
    steps = config.steps if steps is None else steps
    wheelbase = parameters["lf"] + parameters["lr"]
    # Full-lock steering from the source script is not safe at 20 m/s. Keep
    # its smooth signal, but bound the requested lateral acceleration.
    steer_limit = min(
        parameters["s_max"],
        np.arctan(config.lateral_accel_target * wheelbase / speed**2),
    )
    steer_targets = _smooth_steering_targets(steps, config.steering_sines, rng)
    steer_targets *= steer_limit
    speed_targets = np.empty(steps)
    controls = np.empty((steps, 2))
    target_speed = speed
    for index in range(steps):
        if index % config.speed_target_refresh == 0:
            target_speed = float(np.clip(
                speed + rng.uniform(-config.speed_target_span / 2, config.speed_target_span / 2),
                0.5, 20.0,
            ))
        speed_targets[index] = target_speed
        acceleration, steering_speed = _pid(
            target_speed, steer_targets[index], state, parameters
        )
        acceleration += config.controller_noise_scale * truncnorm.rvs(
            -parameters["a_max"], parameters["a_max"], random_state=rng
        )
        steering_speed += config.controller_noise_scale * truncnorm.rvs(
            -parameters["sv_max"], parameters["sv_max"], random_state=rng
        )
        acceleration = np.clip(
            acceleration, -parameters["a_max"],
            min(parameters["a_max"], (20.0 - state[3]) / config.timestep),
        )
        steering_speed = np.clip(
            steering_speed, parameters["sv_min"], parameters["sv_max"]
        )
        controls[index] = [steering_speed, acceleration]
        state = simulator.step(state, controls[index])
        if not np.isfinite(state).all():
            raise ValueError(f"nonfinite nominal probe at {speed:g} m/s, step {index}")
        if abs(state[6]) > config.max_abs_slip or abs(state[5]) > config.max_abs_yaw_rate:
            raise ValueError(f"nominal probe exceeded ST envelope at {speed:g} m/s, step {index}")
        if not -1e-8 <= state[3] <= 20.0 + 1e-8:
            raise ValueError(f"nominal probe exceeded speed range at {speed:g} m/s")
    return {
        "initial_state": _initial(speed),
        "controls": controls,
        "target_speed": speed_targets,
        "target_steering": steer_targets,
    }


def _rollout(simulator, probe, config):
    states = [probe["initial_state"].copy()]
    valid = []
    reason = "step_limit"
    for control in probe["controls"]:
        state = simulator.step(states[-1], control)
        states.append(state)
        supported = (
            np.isfinite(state).all()
            and abs(state[6]) <= config.max_abs_slip
            and abs(state[5]) <= config.max_abs_yaw_rate
            and -1e-8 <= state[3] <= 20.0 + 1e-8
        )
        valid.append(supported)
        if not supported:
            reason = "nonfinite" if not np.isfinite(state).all() else "envelope_limit"
            break
    return np.asarray(states), np.asarray(valid, dtype=bool), reason


def _plot_results(root, config, members, probes, paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    figures = root / "plots"
    figures.mkdir(exist_ok=True)
    colors = plt.get_cmap("viridis")
    norm = Normalize(*config.friction)
    if len(config.speeds) <= 4:
        plot_speeds = config.speeds
    else:
        representative = {
            config.speeds[i]
            for i in np.rint(np.linspace(0, len(config.speeds) - 1, 4)).astype(int)
        }
        peak_speed = max(
            config.speeds,
            key=lambda speed: max(
                np.linalg.norm(
                    paths[(member["instance_id"], f"speed-{speed:g}-signal-0")][-1, :2]
                    - paths[(members[0]["instance_id"], f"speed-{speed:g}-signal-0")][-1, :2]
                )
                for member in members
            ),
        )
        plot_speeds = tuple(speed for speed in config.speeds if speed in representative | {peak_speed})
    fig, axes = plt.subplots(len(plot_speeds), 2, figsize=(14, 3.6 * len(plot_speeds)),
                             squeeze=False, layout="constrained")
    for (ax, deviation_ax), speed in zip(axes, plot_speeds):
        probe_id = f"speed-{speed:g}-signal-0"
        baseline = paths[(members[0]["instance_id"], probe_id)]
        for member in members:
            states = paths[(member["instance_id"], probe_id)]
            ax.plot(states[:, 0], states[:, 1], color=colors(norm(member["friction"])),
                    linewidth=1, alpha=0.55)
            ax.plot(states[-1, 0], states[-1, 1], marker=".",
                    color=colors(norm(member["friction"])), markersize=3)
            count = min(len(states), len(baseline))
            deviation_ax.plot(np.arange(count) * config.timestep,
                              states[:count, 1] - baseline[:count, 1],
                              color=colors(norm(member["friction"])),
                              linewidth=1, alpha=0.55)
        ax.plot(0, 0, "kx", markersize=6)
        ax.set(title=f"Same actions from {speed:g} m/s: paths", xlabel="x [m]", ylabel="y [m]")
        ax.grid(alpha=0.25)
        deviation_ax.axhline(0, color="k", linewidth=0.8)
        deviation_ax.set(title="Lateral position minus baseline", xlabel="time [s]",
                         ylabel="Δy [m]")
        deviation_ax.grid(alpha=0.25)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=axes.ravel().tolist(),
                 label="surface friction μ", shrink=0.65)
    fig.savefig(figures / "paired_trajectories.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    speed = max(config.speeds)
    probe_id = f"speed-{speed:g}-signal-0"
    anchors = {m["anchor"]: m for m in members if m["anchor"] is not None}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    baseline = paths[(anchors["baseline"]["instance_id"], probe_id)]
    for ax, knob in zip(axes.flat, KNOBS):
        for key, label, style in (
            (f"{knob}-low", "low", "C0-"),
            (f"{knob}-high", "high", "C3-"),
        ):
            states = paths[(anchors[key]["instance_id"], probe_id)]
            count = min(len(states), len(baseline))
            deviation = states[:count, 1] - baseline[:count, 1]
            ax.plot(np.arange(count) * config.timestep, deviation, style,
                    label=f"{label} (end {deviation[-1]:+.2f} m)")
        ax.axhline(0, color="k", linewidth=0.8, label="baseline")
        ax.set(title=f"Changing {knob} only; same actions at {speed:g} m/s",
               xlabel="time [s]", ylabel="Δy from baseline [m]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "parameter_effects.png", dpi=170)
    plt.close(fig)

    probe = probes[probe_id]
    times = np.arange(len(probe["controls"])) * config.timestep
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, data, label in zip(
        axes.flat,
        (probe["target_steering"], probe["controls"][:, 0],
         probe["target_speed"], probe["controls"][:, 1]),
        ("Steering target [rad]", "Applied steering speed [rad/s]",
         "Speed target [m/s]", "Applied acceleration [m/s²]"),
    ):
        ax.plot(times, data)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle(f"Shared action tape for every parameter combination ({speed:g} m/s)")
    fig.tight_layout()
    fig.savefig(figures / "shared_actions.png", dpi=170)
    plt.close(fig)


def collect_vehicle(config, emit=print):
    config.validate()
    config = replace(config, seed=resolve_seed(config.seed))
    parameter_seed, probe_seed = np.random.SeedSequence(config.seed).spawn(2)
    parameter_rng = np.random.default_rng(parameter_seed)
    probe_rng = np.random.default_rng(probe_seed)
    instance_specs = parameter_instances(config, parameter_rng)
    per_speed = (
        None if config.transitions_per_instance is None
        else config.transitions_per_instance // len(config.speeds)
    )
    probes = {}
    for speed, signal in product(config.speeds, range(config.signals_per_speed)):
        steps = (
            config.steps if per_speed is None or signal < config.signals_per_speed - 1
            else per_speed - signal * config.steps
        )
        probes[f"speed-{speed:g}-signal-{signal}"] = shared_probe(
            config, speed, probe_rng, steps=steps
        )
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    manifest = {
        "schema_version": DATASET_VERSION,
        "config": asdict(config),
        "state_fields": STATE_FIELDS,
        "control_fields": CONTROL_FIELDS,
        "state_units": ["m", "m", "rad", "m/s", "rad", "rad/s", "rad"],
        "control_units": ["rad/s", "m/s^2"],
        "versions": versions(),
        "instances": [],
        "shard_checksums": {},
        "status": "collecting",
    }
    paths = {}
    total = 0
    incomplete = []
    for index, (values, split, anchor) in enumerate(instance_specs):
        mu, front, rear, mass = map(float, values)
        parameters = vehicle_parameters("fullscale", mu, mass, "fixed")
        parameters["C_Sf"] *= front
        parameters["C_Sr"] *= rear
        simulator = GymST(parameters, config.timestep, config.integrator_timestep)
        manifest["source_fingerprint"] = simulator.source_fingerprint
        member = {
            "instance_id": f"instance-{index:05d}",
            "split": split,
            "anchor": anchor,
            "friction": mu,
            "front_cornering_scale": front,
            "rear_cornering_scale": rear,
            "mass_ratio": mass,
            "parameters": parameters,
            "files": [],
        }
        for probe_id, probe in probes.items():
            states, valid, reason = _rollout(simulator, probe, config)
            count = len(valid)
            filename = f"trajectories/{member['instance_id']}/{probe_id}.npz"
            destination = root / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                states=states,
                controls=probe["controls"][:count],
                commanded_controls=probe["controls"][:count],
                actuator_noise=np.zeros((count, 2)),
                measured_states=states,
                valid=valid,
                times=np.arange(count + 1) * config.timestep,
                termination=np.asarray(reason),
                target_speed=probe["target_speed"][:count],
                target_steering=probe["target_steering"][:count],
            )
            manifest["shard_checksums"][filename] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            member["files"].append(filename)
            paths[(member["instance_id"], probe_id)] = states
            total += int(valid.sum())
            if reason != "step_limit":
                incomplete.append((member["instance_id"], probe_id, reason, count))
        manifest["instances"].append(member)
        write_json(root / "manifest.json", manifest)
        emit(f"[vehicle] {member['instance_id']} | {split} | μ={mu:.3f} | "
             f"Csf×{front:.3f} | Csr×{rear:.3f} | m×{mass:.3f}")

    _plot_results(root, config, manifest["instances"], probes, paths)
    summary = {
        "instances": len(manifest["instances"]),
        "trajectories": len(manifest["instances"]) * len(probes),
        "valid_transitions": total,
        "incomplete_trajectories": incomplete,
        "max_recorded_speed_mps": max(float(s[:, 3].max()) for s in paths.values()),
        "endpoint_spread_m": {},
        "anchor_endpoint_distance_m": {},
    }
    anchors = {m["anchor"]: m for m in manifest["instances"] if m["anchor"] is not None}
    for speed in config.speeds:
        probe_id = f"speed-{speed:g}-signal-0"
        endpoints = np.array([
            paths[(member["instance_id"], probe_id)][-1, :2]
            for member in manifest["instances"]
        ])
        baseline = endpoints[0]
        summary["endpoint_spread_m"][f"{speed:g}"] = float(
            np.linalg.norm(endpoints - baseline, axis=1).max()
        )
        summary["anchor_endpoint_distance_m"][f"{speed:g}"] = {
            name: float(np.linalg.norm(
                paths[(anchor["instance_id"], probe_id)][-1, :2] - baseline
            ))
            for name, anchor in anchors.items() if name != "baseline"
        }
    write_json(root / "summary.json", summary)
    report = [
        "# Paired vehicle ST collection",
        "",
        f"- Instances: {summary['instances']} (nine anchors plus stratified combinations)",
        f"- Trajectories: {summary['trajectories']} ({len(probes)} identical-action probes per instance)",
        f"- Valid transitions: {total}",
        f"- Maximum recorded speed: {summary['max_recorded_speed_mps']:.3f} m/s",
        f"- Incomplete trajectories: {len(incomplete)} (see `summary.json`)",
        "- Yaw inertia held fixed while mass changes; this is a synthetic load approximation.",
        "- These are ST lateral-grip variations; friction does not limit longitudinal acceleration.",
        "",
        "## Maximum endpoint distance from baseline (first signal)",
        "",
    ]
    report += [f"- {speed:g} m/s: {summary['endpoint_spread_m'][f'{speed:g}']:.2f} m"
               for speed in config.speeds]
    report += [
        "", "## One-at-a-time endpoint separation at the highest initial speed", "",
        "| Parameter | Low | High |", "|---|---:|---:|",
    ]
    anchor_distances = summary["anchor_endpoint_distance_m"][f"{max(config.speeds):g}"]
    report += [
        f"| {knob} | {anchor_distances[f'{knob}-low']:.3f} m | "
        f"{anchor_distances[f'{knob}-high']:.3f} m |"
        for knob in KNOBS
    ]
    report += [
        "", "## Plots", "",
        "![Paired trajectories](plots/paired_trajectories.png)", "",
        "![One-at-a-time parameter effects](plots/parameter_effects.png)", "",
        "![Shared action tape](plots/shared_actions.png)", "",
        "The paired-path figure shows representative speeds and the speed with",
        "maximum endpoint spread; the dataset includes all",
        f"{len(config.speeds)} configured speed bands. XY panels use separate x/y scales",
        "so lateral spread is visible; the adjacent",
        "panels show lateral-position difference from the baseline in metres.", "",
        "Each speed uses a different seeded action tape, so the between-speed spread",
        "does not isolate speed alone. All parameter instances at a given speed use",
        "exactly the same actions.", "",
        "The controller uses the smooth random-sinusoid steering targets, 42-step speed-target",
        "refresh, PID law, and truncated-normal control jitter adapted from",
        "`/home/lucerna/Documents/Research/vehicle_data_gen/`. Its output is recorded",
        "once per probe and replayed exactly for every ST parameter combination.",
        "The steering target is additionally capped by a speed-dependent lateral-acceleration",
        "envelope, and the controller jitter is scaled as configured so high-speed",
        "probes remain within the declared ST validity envelope.",
    ]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    manifest["transitions"] = total
    manifest["status"] = "complete"
    manifest["fingerprint"] = fingerprint({
        "config": {k: v for k, v in asdict(config).items() if k != "output"},
        "source": manifest["source_fingerprint"],
        "shards": manifest["shard_checksums"],
    })
    write_json(root / "manifest.json", manifest)
    return root


def replot_vehicle(root):
    """Regenerate plots from saved paired shards without altering the dataset."""
    from dyna_dist.config import VehicleCollectionConfig, load_config
    from dyna_dist.io.dataset import Dataset

    root = Path(root)
    config = load_config(VehicleCollectionConfig, root / "config.yaml")
    dataset = Dataset(root)
    paths = {}
    probes = {}
    for member in dataset.instances:
        for filename in member["files"]:
            probe_id = Path(filename).stem
            shard = dataset.trajectory(filename)
            paths[(member["instance_id"], probe_id)] = shard["states"]
            probes.setdefault(probe_id, {
                "controls": shard["controls"],
                "target_speed": shard["target_speed"],
                "target_steering": shard["target_steering"],
            })
    _plot_results(root, config, dataset.instances, probes, paths)
