"""Independent parameter sweeps with paired excitation and preserved chronology."""

from dataclasses import asdict, replace
import hashlib
import numpy as np
from dyna_dist.adapters.gym_st import GymST, GymEnvironment
from dyna_dist.config import resolve_seed, write_config
from dyna_dist.io.common import fingerprint, new_directory, versions, write_json
from dyna_dist.io.dataset import DATASET_VERSION
from dyna_dist.models.profiles import STATE_FIELDS, CONTROL_FIELDS, vehicle_parameters


def parameter_instances(config, parameter_rng, split_rng):
    pairs = [
        (float(mu), float(mass))
        for mu in np.linspace(*config.friction, config.friction_count)
        for mass in np.linspace(*config.mass_ratio, config.mass_count)
    ]
    pairs += list(
        zip(
            parameter_rng.uniform(*config.friction, config.random_instances),
            parameter_rng.uniform(*config.mass_ratio, config.random_instances),
        )
    )
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) < 3:
        raise ValueError("collection needs at least 3 distinct instances for train/validation/test")
    order = split_rng.permutation(len(pairs))
    train = min(len(pairs) - 2, max(1, int(len(pairs) * config.train_fraction)))
    validation = min(len(pairs) - train - 1, max(1, int(len(pairs) * config.validation_fraction)))
    assignments = {
        int(i): ("train" if rank < train else "validation" if rank < train + validation else "test")
        for rank, i in enumerate(order)
    }
    instances = [(mu, mass, assignments[i]) for i, (mu, mass) in enumerate(pairs)]
    for i in range(config.extrapolation_instances):
        if i % 2:
            mu, mass = config.friction[1] * 1.15, config.mass_ratio[1] * 1.1
        else:
            mu, mass = config.friction[0] * 0.8, config.mass_ratio[0] * 0.9
        instances.append((mu * (1 + 0.01 * (i // 2)), mass, "extrapolation"))
    return instances


def excitation(config, rng, index, parameters):
    state = np.zeros(7, dtype=np.float64)
    state[2] = rng.uniform(*config.steering)
    state[3] = rng.uniform(*config.speed)
    state[5] = rng.uniform(*config.yaw_rate)
    state[6] = rng.uniform(*config.slip)
    low_speed = index >= config.trajectories - round(
        config.trajectories * config.low_speed_fraction
    )
    if low_speed:
        state[3] = rng.uniform(*config.low_speed_range)
        state[6] = np.arctan(
            np.tan(state[2]) * parameters["lr"] / (parameters["lf"] + parameters["lr"])
        )
        state[5] = (
            state[3] * np.cos(state[6]) * np.tan(state[2]) / (parameters["lf"] + parameters["lr"])
        )
    t = np.arange(config.steps)
    controls = np.zeros((config.steps, 2))
    style = index % 4
    if style == 0:
        controls[:, 0] = 0.35 * np.sin(t * 0.18)
        controls[:, 1] = 0.25 * np.sin(t * 0.07)
    elif style == 1:
        controls[:, 0] = np.where((t // 12) % 2, -0.3, 0.3)
        controls[:, 1] = np.where((t // 20) % 2, -0.3, 0.3)
    elif style == 2:
        controls[:, 0] = np.clip(np.sin(t * 0.11) * 0.5, -0.35, 0.35)
        controls[:, 1] = -0.25 if index % 2 else 0.25
    else:
        knots = rng.uniform(-0.4, 0.4, (config.steps // 5 + 2, 2))
        for component in range(2):
            controls[:, component] = np.interp(t, np.arange(len(knots)) * 5, knots[:, component])
    if low_speed:
        # Gym ST can be stiff just above its switch with the BRO-sized RK4 substep.
        # Keep these paired kinematic probes in that branch, not unstable crossing data.
        controls[:, 1] = 0
    return state, controls * np.asarray(config.control_scale)


def collect(config, emit=print):
    config.validate()
    config = replace(config, seed=resolve_seed(config.seed))
    root = new_directory(config.output)
    write_config(config, root / "config.yaml")
    sequences = np.random.SeedSequence(config.seed).spawn(5)
    common_rng, parameter_rng, split_rng, actuator_rng, measurement_rng = (
        np.random.default_rng(s) for s in sequences
    )
    baseline = vehicle_parameters(config.profile)
    common = [excitation(config, common_rng, i, baseline) for i in range(config.trajectories)]
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
    manifest["fingerprint"] = fingerprint(
        {k: v for k, v in asdict(config).items() if k != "output"}
    )
    total = 0
    for index, (mu, mass_ratio, split) in enumerate(
        parameter_instances(config, parameter_rng, split_rng)
    ):
        parameters = vehicle_parameters(config.profile, mu, mass_ratio, config.inertia_policy)
        simulator = GymST(parameters, config.timestep, config.integrator_timestep)
        manifest["source_fingerprint"] = simulator.source_fingerprint
        env = (
            GymEnvironment(parameters, config, int(actuator_rng.integers(2**31)))
            if config.source == "gym_env"
            else None
        )
        member = {
            "instance_id": f"instance-{index:05d}",
            "split": split,
            "friction": float(mu),
            "mass_ratio": float(mass_ratio),
            "parameters": parameters,
            "files": [],
        }
        try:
            for trajectory_index, (initial, commands) in enumerate(common):
                state = initial.copy() if env is None else env.reset(initial)
                states, controls, valid, perturbations = [state], [], [], []
                reason = "step_limit"
                for command in commands:
                    noise = actuator_rng.normal(size=2) * config.actuator_noise
                    control = command + noise
                    if env is None:
                        target = simulator.step(state.copy(), control)
                        terminated, contact = False, False
                    else:
                        target, terminated, contact = env.step(state, control)
                    finite = bool(np.isfinite(target).all())
                    supported = (
                        finite
                        and abs(target[6]) <= config.max_abs_slip
                        and abs(target[5]) <= config.max_abs_yaw_rate
                    )
                    states.append(target)
                    controls.append(control)
                    perturbations.append(noise)
                    # A finite integration explosion is not a usable smooth-ST label.
                    # Keep the raw endpoint, but never let it corrupt training scales.
                    valid.append(supported and not contact)
                    state = target
                    if not finite or contact or terminated:
                        reason = (
                            "nonfinite" if not finite else "contact" if contact else "terminated"
                        )
                        break
                    if not supported:
                        reason = "envelope_limit"
                        break
                states = np.asarray(states)
                filename = (
                    f"trajectories/{member['instance_id']}/trajectory-{trajectory_index:04d}.npz"
                )
                destination = root / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    states=states,
                    controls=np.asarray(controls),
                    commanded_controls=commands[: len(controls)],
                    actuator_noise=np.asarray(perturbations),
                    measured_states=states
                    + measurement_rng.normal(size=states.shape) * config.measurement_noise,
                    valid=np.asarray(valid, dtype=bool),
                    times=np.arange(len(states)) * config.timestep,
                    termination=np.asarray(reason),
                )
                manifest["shard_checksums"][filename] = hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest()
                member["files"].append(filename)
                total += sum(valid)
        finally:
            if env is not None:
                env.close()
        manifest["instances"].append(member)
        write_json(root / "manifest.json", manifest)
        emit(
            f"[collect] {member['instance_id']} | split={split} | mu={mu:.3g} | "
            f"mass={parameters['m']:.3g} kg | transitions={total}"
        )
    manifest["transitions"] = int(total)
    manifest["status"] = "complete"
    manifest["fingerprint"] = fingerprint(
        {
            "config": {k: v for k, v in asdict(config).items() if k != "output"},
            "source": manifest["source_fingerprint"],
            "shards": manifest["shard_checksums"],
        }
    )
    write_json(root / "manifest.json", manifest)
    return root
